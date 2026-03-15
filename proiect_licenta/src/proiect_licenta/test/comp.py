import csv

def comp_files():
    file_path = "Disease_Description.csv"
    diseases = []
    common = []
    nr = 0

    with open(file_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)

        for row in reader:
            disease = row[0].lower()
            diseases.append(disease)
            print(disease)


if __name__ == "__main__":
    comp_files()