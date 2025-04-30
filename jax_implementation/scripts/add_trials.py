import os
import re

# List of scripts to modify
script_files = [
    "run_sweep_eps1_wd0.01.sh",
    "run_sweep_eps1_wd0.001.sh",
    "run_sweep_eps1_wd0.0001.sh",
    "run_sweep_eps3_wd0.01.sh",
    "run_sweep_eps3_wd0.001.sh",
    "run_sweep_eps3_wd0.0001.sh",
    "run_sweep_eps7_wd0.01.sh",
    "run_sweep_eps7_wd0.001.sh",
    "run_sweep_eps7_wd0.0001.sh"
]

script_dir = os.path.dirname(__file__) or "."

for filename in script_files:
    filepath = os.path.join(script_dir, filename)
    if not os.path.exists(filepath):
        print(f"Warning: Script not found, skipping: {filepath}")
        continue

    print(f"Processing {filepath}...")
    with open(filepath, 'r') as f:
        lines = f.readlines()

    new_lines = []
    in_lr_loop = False
    lr_loop_indent = ""
    seed_line_index = -1
    exp_name_line_index = -1
    echo_line_index = -1
    lr_loop_start_index = -1
    lr_loop_end_index = -1

    # First pass to find loop boundaries and target lines
    for i, line in enumerate(lines):
        match = re.match(r"^(\s*)", line)
        current_indent = match.group(1) if match else ""

        # Find the start of the LR loop and its indentation
        if line.strip().startswith("for lr in"):
            if lr_loop_start_index == -1: # Find the first (outer) one
                lr_loop_start_index = i
                lr_loop_indent = current_indent

        # Find the 'done' matching the lr_loop_indent *after* the loop starts
        # This assumes the outer loop's 'done' is the first one encountered
        # at that specific indentation level after the loop starts.
        if lr_loop_start_index != -1 and i > lr_loop_start_index and line.strip() == "done" and current_indent == lr_loop_indent:
            if lr_loop_end_index == -1: # Find the first matching 'done'
                lr_loop_end_index = i
                # We can potentially break here if we are sure this is the correct one
                # break

        # Find python command line index
        python_cmd_index = -1
        if line.strip().startswith("python ../main.py"):
             python_cmd_index = i

        # Find markers relative to the python command line, searching backwards within the loop structure
        if python_cmd_index != -1 and lr_loop_start_index != -1:
             # Search backwards from the python command line up to the start of the LR loop
             search_start = python_cmd_index
             search_end = lr_loop_start_index

             # Find seed line index
             if seed_line_index == -1: # Only search if not already found
                 for j in range(search_start, search_end, -1):
                    # Check if the line contains '--seed' argument part of the python command
                    # Ensure it's formatted as expected (e.g., indented, ends with \)
                    if re.search(r'--seed\s+\d+', lines[j]) and (lines[j].strip().endswith("\\") or lines[j-1].strip().startswith("python")):
                        seed_line_index = j
                        break # Found seed line

             # Find exp_name line index
             if exp_name_line_index == -1:
                 for j in range(search_start, search_end, -1):
                     if lines[j].strip().startswith("exp_name="):
                         exp_name_line_index = j
                         break # Found exp_name line

             # Find echo line index
             if echo_line_index == -1:
                 for j in range(search_start, search_end, -1):
                     if lines[j].strip().startswith("echo \"Running with lr="):
                         echo_line_index = j
                         break # Found echo line

             # Reset python_cmd_index to avoid re-triggering search in same loop iteration
             # if all markers potentially found relative to this python command are checked.
             # This might not be necessary depending on exact loop structure, but can prevent redundant checks.
             # python_cmd_index = -1


    if not (lr_loop_start_index != -1 and lr_loop_end_index != -1 and \
            seed_line_index != -1 and exp_name_line_index != -1 and echo_line_index != -1):
        print(f"  Error: Could not find all necessary markers in {filename}. Skipping.")
        # Print diagnostic info
        # print(f"  Debug: lr_start={lr_loop_start_index}, lr_end={lr_loop_end_index}, seed={seed_line_index}, exp_name={exp_name_line_index}, echo={echo_line_index}")
        continue

    indent = "    " # Indentation for the new trial loop content

    # Construct the new file content
    # Add lines before the LR loop
    new_lines.extend(lines[:lr_loop_start_index])

    # Add the trial loop
    new_lines.append(f"{lr_loop_indent}for trial in {{1..5}}\n")
    new_lines.append(f"{lr_loop_indent}do\n")

    # Process lines within the original LR loop
    for i in range(lr_loop_start_index, lr_loop_end_index + 1):
        line = lines[i]
        new_line_content = line.rstrip('\n')

        # Modify specific lines
        if i == echo_line_index:
            # Add trial number to echo statement
            new_line_content = re.sub(r'(echo "Running with)', f'\1 Trial $trial:', new_line_content)
        elif i == exp_name_line_index:
            # Add trial number to exp_name
            new_line_content = new_line_content.replace('"', '_trial${trial}"')
        elif i == seed_line_index:
             # Modify the seed based on trial number
             # Ensure we target the argument value correctly, handling potential backslash
             has_backslash = new_line_content.strip().endswith("\\")
             base_content = new_line_content.replace('--seed 1024', '--seed $((1024 + trial - 1))')
             # Remove trailing backslash if it exists, modify, then add back if needed
             stripped_content = base_content.strip()
             if has_backslash and stripped_content.endswith("\\"):
                 stripped_content = stripped_content[:-2].strip()

             modified_arg = '--seed $((1024 + trial - 1))'

             # Reconstruct the line, assuming --seed is the only arg or handled by context
             # This simplistic replacement might need adjustment if line structure varies greatly
             parts = new_line_content.split('--seed')
             if len(parts) > 1:
                 prefix = parts[0]
                 # Find space after 1024 to replace up to that point
                 value_part = parts[1].split(None, 1) # Split ' 1024 ...' into ['1024', '...']
                 if len(value_part) > 1:
                     suffix = value_part[1]
                     new_line_content = prefix + modified_arg + ' ' + suffix
                 else: # Handle case where seed is the last argument on the line
                      new_line_content = prefix + modified_arg
             else: # Fallback if split fails unexpectedly
                  new_line_content = new_line_content.replace('1024', '$((1024 + trial - 1))') # Less robust replacement


        # Add indentation for the trial loop
        new_lines.append(f"{lr_loop_indent}{indent}{new_line_content}\n")

    # Add the end of the trial loop
    new_lines.append(f"{lr_loop_indent}done\n")

    # Add lines after the original LR loop end
    new_lines.extend(lines[lr_loop_end_index+1:])

    # Write the modified content back to the file
    with open(filepath, 'w') as f:
        f.writelines(new_lines)
    print(f"  Successfully modified {filename}.")

print("\nScript execution finished.") 