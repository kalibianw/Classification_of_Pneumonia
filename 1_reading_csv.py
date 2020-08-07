import numpy as np
import time
import csv
import os

from utils import DataModule

csv_file_path = "csv/Chest_xray_Corona_Metadata.csv"
dm = DataModule(csv_path=csv_file_path)

os.system("cls")

user_input = int(input("1. train_test_splitted data\n"
                       "2. complete data\n"))

if user_input == 1:
    user_input = int(input("1. Normal or Pnemonia\n"
                           "2. '' or Virus or bacteria\n"
                           "3. '' or ARDS or COVID-19 or SARS or Streptococcus\n"
                           "4. All of above\n"
                           ))

    if user_input == 1:
        dm.reading_csv(2, ["Normal", "Pnemonia"], "Pnemonia", data_augmentation=0)

    elif user_input == 2:
        dm.reading_csv(5, ["", "Virus", "bacteria"], "Virus", data_augmentation=0)

    elif user_input == 3:
        dm.reading_csv(4, ["", "ARDS", "COVID-19", "SARS", "Streptococcus"], "COVID", data_augmentation=0)

    elif user_input == 4:
        dm.reading_csv(2, ["Normal", "Pnemonia"], "Pnemonia", data_augmentation=0)
        dm.reading_csv(5, ["", "Virus", "bacteria"], "Virus", data_augmentation=0)
        dm.reading_csv(4, ["", "ARDS", "COVID-19", "SARS", "Streptococcus"], "COVID", data_augmentation=0)

    else:
        print("Error")

elif user_input == 2:
    user_input = int(input("1. Normal or Pnemonia\n"
                           "2. Virus or bacteria\n"
                           "3. '' or ARDS or COVID-19 or SARS or Streptococcus\n"
                           "4. All of above\n"
                           ))

    if user_input == 1:
        dm.reading_csv(2, ["Normal", "Pnemonia"], "Pnemonia_augmentation_1", data_augmentation=1)

    elif user_input == 2:
        dm.reading_csv(5, ["Virus", "bacteria"], "Virus_bacteria_augmentation_1", data_augmentation=1)

    elif user_input == 3:
        dm.reading_csv(4, ["", "ARDS", "COVID-19", "SARS", "Streptococcus"], "COVID", data_augmentation=1)

    elif user_input == 4:
        dm.reading_csv(2, ["Normal", "Pnemonia"], "Pnemonia", data_augmentation=1)
        dm.reading_csv(5, ["", "Virus", "bacteria"], "Virus", data_augmentation=1)
        dm.reading_csv(4, ["", "ARDS", "COVID-19", "SARS", "Streptococcus"], "COVID", data_augmentation=1)

    else:
        print("Error")

else:
    print("Error")
