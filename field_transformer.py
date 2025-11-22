"""
Field Transformer - Applies mapping transformations from field_mapping_config.json
"""
import json
import re
from pathlib import Path


def load_mapping_config(config_path="field_mapping_config.json"):
    """Load the field mapping configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def flatten_json(json_obj, parent_key='', sep='.'):
    """
    Flatten nested JSON structure with dot notation.
    Example: {"person": {"name": "John"}} -> {"person.name": "John"}
    """
    items = []
    if isinstance(json_obj, dict):
        for k, v in json_obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_json(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to comma-separated strings or keep as list
                if v and all(isinstance(item, (str, int, float, bool)) for item in v):
                    items.append((new_key, ", ".join(str(item) for item in v)))
                elif v and all(isinstance(item, dict) for item in v):
                    # Handle list of objects
                    for i, item in enumerate(v):
                        items.extend(flatten_json(item, f"{new_key}[{i}]", sep=sep).items())
                else:
                    items.append((new_key, v))
            else:
                items.append((new_key, v))
    return dict(items)


def evaluate_expression(expr, data_context):
    """
    Evaluate a simple expression against data context.
    Supports:
    - Direct path: ${path.to.field}
    - Array indexing: .split()[0], .split()[-1], .split()[1:-1]
    - String methods: .strip(), .upper(), .lower()
    - Join operations: ' '.join(list_expression)
    """
    if not expr or not isinstance(expr, str):
        return ""
    
    # Extract variable references like ${path.to.field}
    pattern = r'\$\{([^}]+)\}'
    matches = re.findall(pattern, expr)
    
    if not matches:
        return ""
    
    # Get the full expression inside ${}
    full_expr = matches[0]
    
    # Check if expression starts with a join operation like ' '.join(...)
    join_match = re.match(r"^'([^']*)'\s*\.\s*join\((.+)\)$", full_expr)
    if join_match:
        separator = join_match.group(1)
        inner_expr = join_match.group(2)
        
        # Evaluate the inner expression to get a list
        # Find the path and operations in the inner expression
        method_start = re.search(r'\.(split|strip|upper|lower)\(', inner_expr)
        
        if method_start:
            path = inner_expr[:method_start.start()]
            operations = inner_expr[method_start.start():]
        else:
            path = inner_expr
            operations = ""
        
        # Get value from data context
        value = data_context.get(path, "")
        if value is None or value == "":
            return ""
        
        result = str(value)
        
        # Apply operations to get a list
        if operations:
            try:
                remaining = operations
                while remaining:
                    if remaining.startswith('.split()'):
                        parts = result.split()
                        remaining = remaining[8:]
                        
                        # Check for slice notation [start:end]
                        slice_match = re.match(r'\[(-?\d+)?:(-?\d+)?\]', remaining)
                        if slice_match:
                            start_str = slice_match.group(1)
                            end_str = slice_match.group(2)
                            start = int(start_str) if start_str else None
                            end = int(end_str) if end_str else None
                            parts = parts[start:end]
                            remaining = remaining[len(slice_match.group(0)):]
                        
                        result = parts
                        break
                    else:
                        break
            except Exception:
                return ""
        else:
            # No operations, try to split by default
            result = result.split()
        
        # Join with separator
        if isinstance(result, list):
            return separator.join(result)
        else:
            return str(result)
    
    # Original logic for non-join expressions
    # Find the first method call
    method_start = re.search(r'\.(split|strip|upper|lower)\(', full_expr)
    
    if method_start:
        # Split into path and operations
        path = full_expr[:method_start.start()]
        operations = full_expr[method_start.start():]
    else:
        # Check for array indexing without methods
        array_match = re.search(r'\[(-?\d+)\]', full_expr)
        if array_match:
            path = full_expr[:array_match.start()]
            operations = full_expr[array_match.start():]
        else:
            # Simple path with no operations
            path = full_expr
            operations = ""
    
    # Get value from data context
    value = data_context.get(path, "")
    
    if value is None or value == "":
        return ""
    
    # Convert to string for operations
    result = str(value)
    
    # Apply operations sequentially
    if operations:
        try:
            # Parse and apply operations in order
            remaining = operations
            
            while remaining:
                # Check for .split()
                if remaining.startswith('.split()'):
                    parts = result.split()
                    remaining = remaining[8:]  # Remove '.split()'
                    
                    # Check if followed by array indexing
                    index_match = re.match(r'\[(-?\d+)\]', remaining)
                    if index_match:
                        index = int(index_match.group(1))
                        if 0 <= index < len(parts):
                            result = parts[index]
                        elif index < 0 and abs(index) <= len(parts):
                            result = parts[index]
                        else:
                            result = ""
                        remaining = remaining[len(index_match.group(0)):]
                    else:
                        # No index, join back
                        result = " ".join(parts)
                
                # Check for .strip()
                elif remaining.startswith('.strip()'):
                    result = result.strip()
                    remaining = remaining[8:]
                
                # Check for .upper()
                elif remaining.startswith('.upper()'):
                    result = result.upper()
                    remaining = remaining[8:]
                
                # Check for .lower()
                elif remaining.startswith('.lower()'):
                    result = result.lower()
                    remaining = remaining[8:]
                
                # Check for direct array indexing [index]
                elif remaining.startswith('['):
                    index_match = re.match(r'\[(-?\d+)\]', remaining)
                    if index_match:
                        index = int(index_match.group(1))
                        # Treat as character or list indexing
                        if isinstance(result, list):
                            if 0 <= index < len(result):
                                result = result[index]
                            elif index < 0 and abs(index) <= len(result):
                                result = result[index]
                            else:
                                result = ""
                        else:
                            # String indexing
                            result_str = str(result)
                            if 0 <= index < len(result_str):
                                result = result_str[index]
                            elif index < 0 and abs(index) <= len(result_str):
                                result = result_str[index]
                            else:
                                result = ""
                        remaining = remaining[len(index_match.group(0)):]
                    else:
                        break
                else:
                    # Unknown operation, stop processing
                    break
        
        except Exception as e:
            # If any operation fails, return empty string
            result = ""
    
    return result.strip() if result else ""


def apply_mapping(merged_json, mapping_config=None):
    """
    Apply field mapping configuration to transform extracted JSON to CSV format.
    
    Args:
        merged_json: The merged JSON data from extraction
        mapping_config: Optional mapping config (loads from file if not provided)
    
    Returns:
        Dictionary with PropertyName as keys and transformed values
    """
    if mapping_config is None:
        mapping_config = load_mapping_config()
    
    # Flatten the JSON with dot notation
    flat_data = flatten_json(merged_json)
    
    # Initialize result with all property names
    result = {}
    
    # Apply each mapping rule
    for rule in mapping_config:
        property_name = rule.get("PropertyName", "")
        from_expr = rule.get("from", "")
        
        # Evaluate the expression
        value = evaluate_expression(from_expr, flat_data)
        
        result[property_name] = value
    
    return result


def get_property_names_in_order():
    """Get all property names in the order defined in mapping config."""
    mapping_config = load_mapping_config()
    return [rule.get("PropertyName", "") for rule in mapping_config]


def get_mapping_by_property(property_name):
    """Get mapping rule for a specific property name."""
    mapping_config = load_mapping_config()
    for rule in mapping_config:
        if rule.get("PropertyName") == property_name:
            return rule
    return None


def preview_mapping(merged_json, limit=10):
    """
    Preview the mapping transformation for debugging.
    Returns list of tuples: (PropertyName, from_expr, value)
    """
    mapping_config = load_mapping_config()
    flat_data = flatten_json(merged_json)
    
    results = []
    for rule in mapping_config[:limit]:
        property_name = rule.get("PropertyName", "")
        from_expr = rule.get("from", "")
        value = evaluate_expression(from_expr, flat_data)
        
        results.append({
            "property": property_name,
            "expression": from_expr,
            "value": value,
            "filled": bool(value and value.strip())
        })
    
    return results
