from sklearn.metrics import classification_report

# Process dataset label to get 1 or 0
def process_dataset_label(s):
    s = s.strip()
    if s == "supports":
        return 1
    else:   # s == "refutes"
        return 0

# Process model generation to get 1 or 0
def process_model_generation(s):
    s = s.strip().lower()
    true_index = s.find("true")
    false_index = s.find("false")

    if true_index == -1 and false_index == -1:
        return 0
    elif true_index != -1 and (false_index == -1 or true_index < false_index):
        return 1
    else:
        return 0

# Open log.txt and process each line to get "Expected" and "Answer" values
def read_log():
    with open("log.txt", "r") as f:
        lines = f.readlines()
    # If line doesn't start with "Questions: ", merge it with previous line
    for i in range(len(lines)-1, 1, -1):
        if not lines[i].startswith("Question: "):
            lines[i-1] = lines[i-1].strip() + " " + lines[i].strip()
            lines[i] = ""
    lines = [line for line in lines if line != ""]
    expected = []
    answer = []
    for line in lines:
        # if (not line.startswith("Questions: ")) or ("Expected: " not in line) or ("Answer: " not in line):
        #     print("Invalid line: " + line)
        #     continue
        e = line.split("Expected: ")[1].split("|")[0]
        a = line.split("Answer: ")[1].split("|")[0]
        expected.append(process_dataset_label(e))
        answer.append(process_model_generation(a))
    return expected, answer

# Compute confusion matrix using sklearn
def compute_confusion_matrix(expected, answer):
    cm = classification_report(expected, answer)
    return cm

# Save result to file
def write_confusion_matrix_to_file(cm):
    with open("result(2).txt", "w") as f:
        f.write(str(cm))
    f.close()

expected, answer = read_log()
cm = compute_confusion_matrix(expected, answer)
write_confusion_matrix_to_file(cm)
