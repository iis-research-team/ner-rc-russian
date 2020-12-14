import xlrd
import csv
import os


def csv_from_excel(inp, out, sheet):
    wb = xlrd.open_workbook(inp, encoding_override="utf-8")
    sh = wb.sheet_by_name(sheet)
    your_csv_file = open(out, 'w', encoding="utf-8")
    wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)

    for rownum in range(sh.nrows):
        wr.writerow(sh.row_values(rownum))

    your_csv_file.close()


if __name__ == "__main__":
    dir_names = [d for d in os.listdir("annotated_files") if d not in ["part_8", "Relations.xlsx", "relations_diff.xlsx"]]

    if not os.path.exists("raw_markup"):
        os.makedirs("raw_markup")

    for d in dir_names:
        file_names = os.listdir(os.path.join("annotated_files", d))
        for file in file_names:
            name = file[:file.find(".")]
            print(file, name)
            csv_from_excel(os.path.join("annotated_files", d, file),
                           os.path.join("raw_markup", name + ".csv"),
                           name)
    csv_from_excel(os.path.join("annotated_files", "Relations.xlsx"),
                   os.path.join("raw_markup", "Relations.csv"),
                   "Sheet1")
