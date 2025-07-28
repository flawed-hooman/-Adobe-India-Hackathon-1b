import json
import os
import fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import datetime
import re
from collections import defaultdict

TEXT_FONT_BOLD = 16
TEXT_FONT_ITALIC = 2


def normalize(text):
    return re.sub(r"\s{2,}", " ", text.strip())

def is_text_bold(flags):
    return (flags & TEXT_FONT_BOLD) != 0

def is_text_italic(flags):
    return (flags & TEXT_FONT_ITALIC) != 0

def is_heading_candidate(text):
    return len(text.strip()) >= 3 and re.search(r"[A-Za-z]", text)

def is_noise_text(text):
    """
    Filters out common boilerplate, page numbers, dates, addresses, and other non-heading text
    using more general patterns.
    """
    bad_keywords = ["copyright", "version", "address", "rsvp", "phone", "email", "website", "date", "time", "fax", "tel", "licence", "disclaimer", "acknowledgement", "appendix"]
    text_lower = text.lower()

    if any(bad in text_lower for bad in bad_keywords):
        return True

    if re.fullmatch(r"^\s*page\s+\d+\s*$", text_lower) or \
       re.fullmatch(r"^\s*\d+\s+of\s+\d+\s*$", text_lower) or \
       re.fullmatch(r"^\s*\d+\s*$", text_lower):
        return True

    if re.search(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s+\d{4}\b", text_lower) or \
       re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", text_lower) or \
       re.search(r"\b\d{4}-\d{2}-\d{2}\b", text_lower):
        return True

    if re.search(r"\b\d{5}(-\d{4})?\b", text) or \
       re.search(r"\b(?:road|street|avenue|pkwy|parkway|drive|blvd|lane|sq|terrace|circ|on|ky|ny|ca|tx)\b", text_lower):
        return True

    if re.search(r"(.)\1{3,}", text):
        return True

    if len(text.strip()) < 3 and not re.search(r"[A-Za-z0-9]{2,}", text):
        return True
    if len(text.split()) > 18 :
        return True
    if text.strip().endswith((".", ";", ",")) and not re.match(r"^\d+(\.\d+)*[\.\s]*", text.strip()):
        return True

    if text_lower.startswith("result:"):
        return True

    return False

def is_list_item_start(text):
    """
    Checks if a text line starts with common list item markers.
    """
    
    list_patterns = [
        r"^\s*[-*•]\s+", # Bullet points
        r"^\s*\d+\.\s+", # Numbered list (e.g., "1. ")
        r"^\s*\(\d+\)\s+", # Numbered list (e.g., "(1) ")
        r"^\s*[a-zA-Z]\.\s+", # Lettered list (e.g., "a. ")
        r"^\s*\([a-zA-Z]\)\s+", # Lettered list (e.g., "(a) ")
    ]
    for pattern in list_patterns:
        if re.match(pattern, text):
            return True
    return False

def cluster_spans_by_line(spans, y_thresh=2.0, x_thresh=5.0):
    lines = defaultdict(list)
    for span in spans:
        y = round(span["bbox"][1], 1)
        matched_y = None
        for line_y in lines:
            if abs(line_y - y) < y_thresh:
                matched_y = line_y
                break
        key = matched_y if matched_y is not None else y
        lines[key].append(span)

    merged = []
    for y, spans_in_line in lines.items():
        spans_in_line = sorted(spans_in_line, key=lambda s: s["bbox"][0])
        words = []
        last_x_end = None
        font_size = 0
        is_bold = False
        is_italic = False
        for span in spans_in_line:
            if last_x_end is not None and (span["bbox"][0] - last_x_end > x_thresh):
                words.append(" ")
            words.append(span["text"])
            last_x_end = span["bbox"][2]
            font_size = max(font_size, round(span["size"]))
            if is_text_bold(span["flags"]):
                is_bold = True
            if is_text_italic(span["flags"]):
                is_italic = True
        text = normalize("".join(words))
        merged.append({
            "text": text,
            "font_size": font_size,
            "y": y,
            "x": spans_in_line[0]["bbox"][0],
            "page": spans_in_line[0]["number"],
            "is_bold": is_bold,
            "is_italic": is_italic
        })
    return sorted(merged, key=lambda l: (l['page'], l['y']))

def group_lines_into_blocks(lines, y_gap_thresh=30.0):
    blocks = []
    current_block = []
    for i, line in enumerate(lines):
        if not current_block:
            current_block.append(line)
            continue
        last_line = current_block[-1]
        same_page = last_line['page'] == line['page']
        same_font = last_line['font_size'] == line['font_size']
        style_match = last_line['is_bold'] == line['is_bold'] and last_line['is_italic'] == line['is_italic']
        y_dist = abs(line['y'] - last_line['y'])
        if same_page and same_font and style_match and y_dist < y_gap_thresh:
            current_block.append(line)
        else:
            blocks.append(current_block)
            current_block = [line]
    if current_block:
        blocks.append(current_block)
    return blocks

def get_block_score(block):
    score = 0
    avg_size = sum([l["font_size"] for l in block]) / len(block)
    if len(block) <= 2:
        score += 2
    if any(l["is_bold"] for l in block):
        score += 2
    if any(l["is_italic"] for l in block):
        score += 1
    score += avg_size * 0.2

    full_text = normalize(" ".join(l["text"] for l in block))
    if len(full_text.split()) > 18:
        score -= 3

    return score

def get_heading_level(score, max_score):
    if max_score == 0:
        return "H3"

    relative_score = score / max_score
    if relative_score >= 0.9:
        return "H1"
    elif relative_score >= 0.7:
        return "H2"
    else:
        return "H3"

# --- Tree Node Structure ---
class Node:
    def __init__(self, type, text, page=None, level=None):
        self.type = type # "header" or "paragraph"
        self.text = text
        self.page = page
        self.level = level # Only for headers (e.g., "H1", "H2", "H3")
        self.children = [] # Only for headers

    def add_child(self, child_node):
        self.children.append(child_node)

    def to_dict(self):
        result = {
            "type": self.type,
            "text": self.text,
            "page": self.page
        }
        if self.type == "header":
            result["level"] = self.level
            result["children"] = [child.to_dict() for child in self.children]
        return result

# --- Core Logic for Tree Construction ---

def build_content_tree(pdf_path):
    """
    Extracts content from PDF, identifies headers/paragraphs, and builds a tree structure.
    """
    doc = fitz.open(pdf_path)
    all_lines = []
    for i, page in enumerate(doc):
        try:
            page_data = json.loads(page.get_text("json"))
            spans = []
            for block in page_data.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            span["number"] = i 
                            spans.append(span)
            if spans:
                lines = cluster_spans_by_line(spans)
                lines = [l for l in lines if not (
                    l['x'] > 0.8 * doc[l['page']].rect.width or
                    l['y'] > 0.9 * doc[l['page']].rect.height or
                    re.match(r"confidential|draft", l["text"], re.IGNORECASE) or
                    (len(l["text"]) < 5 and not re.search(r"[A-Za-z0-9]", l["text"]))
                )]
                all_lines.extend(lines)
        except Exception as e:
            print(f"Error processing page {i+1} of {os.path.basename(pdf_path)}: {e}")
            continue

    if not all_lines:
        return []

    blocks = group_lines_into_blocks(all_lines)

    block_scores = []
    for block in blocks:
        block_text = normalize(" ".join(l["text"] for l in block))
        if is_heading_candidate(block_text) and not is_noise_text(block_text):
            block_scores.append(get_block_score(block))
        else:
            block_scores.append(0)

    max_score = max(block_scores) if block_scores else 0

    root_nodes = []
    header_stack = []

    def level_to_numeric(level_str):
        if level_str == "H1": return 1
        if level_str == "H2": return 2
        if level_str == "H3": return 3
        return 99

    for i, block_lines in enumerate(blocks):
        block_text = normalize(" ".join(l["text"] for l in block_lines))
        current_page = block_lines[0]["page"] + 1

        is_header_block = False
        header_level = None

        # Determine if it's a header candidate based on score and filters
        if is_heading_candidate(block_text) and not is_noise_text(block_text) and block_scores[i] > (max_score * 0.4):
            if is_list_item_start(block_text):
                is_header_block = False
            else:
                is_header_block = True
                header_level = get_heading_level(block_scores[i], max_score)

        if is_header_block:
            current_numeric_level = level_to_numeric(header_level)
            header_node = Node(type="header", text=block_text, page=current_page, level=header_level)

            while header_stack and level_to_numeric(header_stack[-1][0]) >= current_numeric_level:
                header_stack.pop()

            if header_stack:
                header_stack[-1][1].add_child(header_node)
            else:
                root_nodes.append(header_node)

            header_stack.append((header_level, header_node))

        else: 
            paragraph_node = Node(type="paragraph", text=block_text, page=current_page)
            if header_stack:
                header_stack[-1][1].add_child(paragraph_node)
            else:
                root_nodes.append(paragraph_node)

    return [node.to_dict() for node in root_nodes]

# --- General Utility Functions ---

def process_all_pdfs_into_trees(input_dir, output_dir, document_list_from_input):
    """
    Processes all PDFs in the input_dir and generates a tree structure JSON for each.
    Only processes PDFs specified in the input JSON document list.
    """
    os.makedirs(output_dir, exist_ok=True)

    processed_trees_data = {}

    for doc_info in document_list_from_input:
        filename = doc_info["document_name"]
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            if not os.path.exists(pdf_path):
                print(f"Warning: PDF not found at {pdf_path}. Skipping.")
                continue

            print(f"Building content tree for {filename}...")
            try:
                tree_structure = build_content_tree(pdf_path)
                out_path = os.path.join(output_dir, filename.replace(".pdf", ".json"))
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(tree_structure, f, indent=2, ensure_ascii=False)
                print(f"Successfully created tree for {filename} at {out_path}")
                processed_trees_data[filename] = tree_structure
            except Exception as e:
                print(f"Error building tree for {filename}: {e}")
    return processed_trees_data

def get_current_kolkata_time():
    """Returns the current timestamp in Kolkata timezone, formatted as ISO 8601."""
    utc_now = datetime.datetime.utcnow()
    kolkata_offset = datetime.timedelta(hours=5, minutes=30)
    kolkata_time = utc_now + kolkata_offset
    return kolkata_time.isoformat(timespec='milliseconds')

def smart_truncate(text, max_chars):
    """Truncates text to max_chars, trying to cut at a word boundary and adding ellipsis."""
    if len(text) <= max_chars:
        return text.strip()

    truncated_text = text[:max_chars]
    last_space = truncated_text.rfind(' ')
    if last_space != -1 and last_space > max_chars * 0.8:
        return truncated_text[:last_space].strip() + "..."
    return truncated_text.strip() + "..."

# --- Main Execution Logic ---
def main():
    base_input_dir = os.path.join(os.getcwd(), "input")
    base_output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(base_output_dir, exist_ok=True)
    collections = [d for d in os.listdir(base_input_dir) if os.path.isdir(os.path.join(base_input_dir, d))]
    
    if not collections:
        print("No collection folders found")
        return

    for coll in sorted(collections):
        coll_input_path = os.path.join(base_input_dir, coll)
        coll_output_path = os.path.join(base_output_dir, f"output_{coll}.json")

        pdf_dir = os.path.join(coll_input_path, "PDFs")
        input_json_path = os.path.join(coll_input_path, "input.json")

        if not os.path.isdir(pdf_dir):
            print(f"PDFs folder missing for collection: {coll}, skipping.")
            continue
        if not os.path.isfile(input_json_path):
            print(f"input.json missing for collection: {coll}, skipping.")
            continue
        
        print(f"Processing collection: {coll}")

        with open(input_json_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        # Run the analysis on this collection
        processed_trees = process_all_pdfs_into_trees(pdf_dir, coll_input_path, input_data["documents"])

        if not processed_trees:
            print("No PDF trees were built. Exiting.")
            return

        model = SentenceTransformer('all-MiniLM-L6-v2')
        dimension = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dimension)

        # --- Populate extracted_sections ---
        all_headers_flat = []
        for doc_name, tree_root_nodes in processed_trees.items():
            def collect_all_nodes(nodes, current_doc_name):
                collected_items = []
                for node in nodes:
                    collected_items.append({
                        "document": current_doc_name,
                        "section_title": node['text'],
                        "page_number": node['page'],
                        "content_type": node['type'],
                        "content": node['text']
                    })
                    if node['type'] == 'header' and node.get('children'):
                        collected_items.extend(collect_all_nodes(node['children'], current_doc_name))
                return collected_items
            all_headers_flat.extend(collect_all_nodes(tree_root_nodes, doc_name))

        corpus_for_embedding = [item for item in all_headers_flat if item["content_type"] == "header" and item["content"].strip()]

        if not corpus_for_embedding:
            print("No header content available for embedding and ranking for extracted_sections. Falling back to page-level content.")
            all_pages_flat = []
            for doc_info in input_data["documents"]:
                pdf_name = doc_info["document_name"]
                pdf_path = os.path.join(pdf_dir, pdf_name)
                if os.path.exists(pdf_path):
                    doc = fitz.open(pdf_path)
                    for i, page in enumerate(doc):
                        text = page.get_text()
                        if text.strip():
                            all_pages_flat.append({
                                "document": pdf_name,
                                "section_title": f"Page {i+1} content",
                                "page_number": i + 1,
                                "content_type": "page",
                                "content": text.strip()
                            })
            corpus_for_embedding = all_pages_flat

        if not corpus_for_embedding:
            print("No content available for embedding and ranking. Exiting RAG part.")
            return

        texts_to_embed = [item["content"] for item in corpus_for_embedding]
        embeddings = model.encode(texts_to_embed, show_progress_bar=True).astype('float32')
        index.add(embeddings)

        master_query = f"As a {input_data['persona']}, {input_data['job_to_be_done']}"
        query_embedding = model.encode([master_query]).astype('float32')

        distances, indices = index.search(query_embedding, 10)

        extracted_sections_output = []
        added_extracted_sections_keys = set()
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            original_item = corpus_for_embedding[idx]
            key = (original_item["document"], original_item["page_number"], original_item["section_title"])
            if key not in added_extracted_sections_keys:
                extracted_sections_output.append({
                    "document": original_item["document"],
                    "section_title": original_item["section_title"],
                    "importance_rank": len(extracted_sections_output) + 1,
                    "page_number": original_item["page_number"]
                })
                added_extracted_sections_keys.add(key)
                if len(extracted_sections_output) >= 5: break

        subsection_analysis_output = []
        subsection_added_keys = set()
        SUBSECTION_CHAR_LIMIT = 400

        job_keywords_phrase = input_data["job_to_be_done"].lower()
        relevant_subtopics = ["activities", "nightlife", "cuisine", "restaurants", "hotels", "tips", "tricks", "packing", "cities"]

        candidate_for_subsection = []
        for doc_name, tree_root_nodes in processed_trees.items():
            def collect_paragraph_text_under_header(nodes):
                """Recursively collects all paragraph text under a given list of nodes."""
                combined_text_parts = []
                for node in nodes:
                    if node['type'] == 'paragraph' and node['text'].strip():
                        combined_text_parts.append(node['text'])
                    elif node['type'] == 'header' and node.get('children'):
                        # Recursively collect from children headers
                        combined_text_parts.append(collect_paragraph_text_under_header(node['children']))
                return " ".join(combined_text_parts).strip()

            # Iterate through top-level nodes to find headers or significant paragraphs
            for node in tree_root_nodes:
                if node['type'] == 'header':
                    # For a header, concatenate all its child paragraphs
                    combined_content = collect_paragraph_text_under_header(node['children'])
                    if combined_content and len(combined_content) > 50: # Only add if substantial content
                        is_relevant = False
                        node_text_lower = node['text'].lower()
                        # Check relevance based on header text itself
                        if any(keyword in node_text_lower for keyword in relevant_subtopics):
                            is_relevant = True

                        # Also check if any part of the combined content is relevant to job_to_be_done
                        if not is_relevant and any(phrase in combined_content.lower() for phrase in job_keywords_phrase.split()):
                            is_relevant = True

                        if is_relevant:
                            candidate_for_subsection.append({
                                "document": doc_name,
                                "full_content": node['text'] + "\n" + combined_content, # Header text + combined paragraphs
                                "page_number": node['page'],
                                "node_type": "header_with_paragraphs" # Custom type for clarity
                            })
                elif node['type'] == 'paragraph' and node['text'].strip() and len(node['text']) > 50:
                    # If a significant standalone paragraph, consider it too
                    node_text_lower = node['text'].lower()
                    is_relevant = False
                    if any(keyword in node_text_lower for keyword in relevant_subtopics) or \
                    any(phrase in node_text_lower for phrase in job_keywords_phrase.split()):
                        is_relevant = True
                    if is_relevant:
                        candidate_for_subsection.append({
                            "document": doc_name,
                            "full_content": node['text'],
                            "page_number": node['page'],
                            "node_type": "paragraph"
                        })

        if candidate_for_subsection:
            subsection_texts_to_embed = [item["full_content"] for item in candidate_for_subsection]
            subsection_embeddings = model.encode(subsection_texts_to_embed, show_progress_bar=True).astype('float32')
            subsection_index = faiss.IndexFlatL2(dimension)
            subsection_index.add(subsection_embeddings)

            subsection_distances, subsection_indices = subsection_index.search(query_embedding, 15)

            for i, idx in enumerate(subsection_indices[0]):
                if idx == -1: continue
                original_candidate = candidate_for_subsection[idx]
                key = (original_candidate["document"], original_candidate["page_number"], original_candidate["full_content"][:50])

                if key not in subsection_added_keys:
                    refined_text = smart_truncate(original_candidate["full_content"], SUBSECTION_CHAR_LIMIT)

                    if len(refined_text) > 50:
                        subsection_analysis_output.append({
                            "document": original_candidate["document"],
                            "refined_text": refined_text,
                            "page_number": original_candidate["page_number"]
                        })
                        subsection_added_keys.add(key)
                        if len(subsection_analysis_output) >= 5:
                            break

        # --- Generate final output similar to challenge1b_output.json structure ---
        final_output_structure = {
            "metadata": {
                "input_documents": [d["document_name"] for d in input_data["documents"]],
                "persona": input_data["persona"],
                "job_to_be_done": input_data["job_to_be_done"],
                "processing_timestamp": get_current_kolkata_time()
            },
            "extracted_sections": extracted_sections_output,
            "subsection_analysis": subsection_analysis_output
        }
        
        with open(coll_output_path, "w", encoding="utf-8") as f:
                json.dump(final_output_structure, f, indent=4, ensure_ascii=False)
        print(f"Saved output for {coll} to {coll_output_path}\n")

    print("\nScript finished successfully.")

if __name__ == "__main__":
    main()