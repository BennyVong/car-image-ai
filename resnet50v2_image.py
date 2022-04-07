from unittest import loader
from sklearn import datasets
from torch.utils.data import DataLoader
from PIL import Image
import torch
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
from dataset_make import construct_ds, show_batch
from model import TransferModel
from tensorflow.keras.optimizers import Adam
import pandas as pd

files = [file for file in os.listdir('./combinations/class') if file.endswith(".jpg")]
file_paths = ['./combinations/class/' + file for file in files]
car_brands = ["Acura", "Alfa Romeo", "Aston Martin", "Audi", "Bentley", "BMW", "Buick", "Cadillac", "Chevrolet", "Chrysler", "Dodge", "Ferrari", "FIAT", "Ford", "Genesis", "GMC", "Honda", "Hyundai", "INFINITI", "Jaguar", "Jeep", "Kia", "Lamborghini", "Land Rover", "Lexus", "Lincoln", "Maserati", "Mazda", "McLaren", "Mercedes-Benz", "MINI", "Mitsubishi", "Nissan", "Porsche", "Ram", "Rolls-Royce", "smart", "Subaru", "Tesla", "Toyota", "Volkswagen", "Volvo"]
car_makes = ["ILX", "MDX", "NSX", "RDX", "RLX", "TLX", "4C Spider", "4C", "Giulia", "Stelvio", "DB11", "DBS", "Vanquish", "Vantage", "A3", "A4", "A5", "A6", "A7", "A8", "e-tron", "Q3", "Q5", "Q7", "Q8", "R8", "TT", "Bentayga", "Continental GT", "Flying Spur", "Mulsanne", "2-Series", "3-Series", "4-Series", "5-Series", "6-Series", "7-Series", "8-Series", "i3", "i8", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "Z4", "Cascada", "Enclave", "Encore", "Envision", "Lacrosse", "Regal", "ATS", "CT4", "CT5", "CT6", "CTS", "Escalade", "XT4", "XT5", "XT6", "XTS", "Blazer", "Bolt EV", "Camaro", "Colorado", "Corvette", "Cruze", "Equinox", "Impala", "Malibu", "Silverado 1500", "Silverado 2500HD", "Sonic", "Spark", "Suburban", "Tahoe", "TrailBlazer", "Traverse", "Trax", "Volt", "300", "Pacifica", "Challenger", "Charger", "Durango", "Grand Caravan", "Journey", "488 GTB", "GTC4Lusso", "Portofino", "124 Spider", "500e", "500L", "500X", "500", "Ecosport", "Edge", "Escape", "Expedition", "Explorer", "F-150", "Fiesta", "Flex", "Fusion", "Mustang", "Ranger", "Super Duty F-250", "Taurus", "Transit Connect Wagon", "G70", "G80", "G90", "Acadia", "Canyon", "Sierra 1500", "Sierra 2500HD", "Terrain", "Yukon", "Accord", "Civic", "Clarity", "CR-V", "Fit", "HR-V", "Insight", "Odyssey", "Passport", "Pilot", "Ridgeline", "Accent", "Elantra", "Ioniq", "Kona Electric", "Kona", "NEXO", "Palisade", "Santa Fe", "Sonata", "Tucson", "Veloster", "Venue", "Q50", "Q60", "Q70", "QX30", "QX50", "QX60", "QX80", "E-Pace", "F-Pace", "F-Type", "I-Pace", "XE", "XF", "XJ", "Cherokee", "Compass", "Gladiator", "Grand Cherokee", "Renegade", "Wrangler", "Cadenza", "Forte", "K900", "Niro", "Optima", "Rio", "Sedona", "Sorento", "Soul EV", "Soul", "Sportage", "Stinger", "Telluride", "Aventador", "Huracan", "Urus", "Defender", "Discovery Sport", "Discovery", "Range Rover Evoque", "Range Rover Sport", "Range Rover Velar", "Range Rover", "ES", "GS", "GX", "IS", "LC", "LS", "LX", "NX", "RC", "RX", "UX", "Aviator", "Continental", "Corsair", "MKC", "MKT", "MKZ", "Nautilus", "Navigator", "Ghibli", "GranTurismo", "Levante", "Quattroporte", "CX-30", "CX-3", "CX-5", "CX-9", "Mazda3 Hatchback", "MAZDA3", "MAZDA6", "MX-5 Miata", "570GT", "570S", "720S", "A Class", "AMG GT", "C Class", "CLA Class", "CLS Class", "E Class", "EQC", "G Class", "GLA Class", "GLB Class", "GLC Class", "GLE Class", "GLS Class", "Metris", "S Class", "SL Class", "SLC Class", "Clubman", "Cooper Countryman", "Cooper", "Eclipse Cross", "Mirage", "Outlander Sport", "Outlander", "370Z", "Altima", "Armada", "Frontier", "GT-R", "Kicks", "Leaf", "Maxima", "Murano", "NV200", "Pathfinder", "Rogue Sport", "Rogue", "Sentra", "Titan", "Versa", "718 Spyder", "718", "911", "Cayenne", "Macan", "Panamera", "Taycan", "1500", "2500", "Cullinan", "Dawn", "Ghost", "Phantom", "Wraith", "fortwo", "Ascent", "BRZ", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "STI S209", "WRX", "Model 3", "Model S", "Model X", "Model Y", "4Runner", "86", "Avalon", "C-HR", "Camry", "Corolla", "Highlander", "Land Cruiser", "Mirai", "Prius C", "Prius", "RAV4", "Sequoia", "Sienna", "Supra", "Tacoma", "Tundra", "Yaris Hatchback", "Yaris", "Arteon", "Atlas"]
car_combinations = ['Ford_Expedition', 'BMW_i8', 'Toyota_Avalon', 'Chevrolet_Blazer', 'Ford_Edge', 'Lexus_LS', 'Lamborghini_Urus', 'Acura_RLX', 'Nissan_Titan', 'Chevrolet_Traverse', 'Hyundai_Veloster', 'Ford_Flex', 'BMW_X7', 'Buick_Encore', 'Maserati_GranTurismo', 'Toyota_Camry', 'Volkswagen_Beetle', 'Chevrolet_Corvette', 'Mercedes-Benz_A Class', 'Dodge_Durango', 'Rolls-Royce_Phantom', 'Jeep_Gladiator', 'Alfa Romeo_Giulia', 'Mercedes-Benz_CLA Class', 'Mercedes-Benz_GLA Class', 'Cadillac_CT5', 'Chevrolet_Impala', 'Nissan_Leaf', 'Bentley_Continental GT', 'Audi_e-tron', 'Acura_NSX', 'Ram_1500', 'GMC_Yukon', 'Jeep_Wrangler', 'Lincoln_MKZ', 'Buick_Envision', 'Aston Martin_Vantage', 'Subaru_Impreza', 'INFINITI_Q60', 'Nissan_Rogue Sport', 'Genesis_G70', 'INFINITI_QX80', 'Mercedes-Benz_EQC', 'Lexus_RX', 'Porsche_Cayenne', 'Kia_Forte', 'Honda_HR-V', 'Nissan_GT-R', 'Subaru_BRZ', 'Kia_Telluride', 'Volkswagen_Jetta', 'Kia_Cadenza', 'Acura_RDX', 'Honda_Civic', 'Honda_CR-V', 'Land Rover_Range Rover Velar', 'Mercedes-Benz_GLC Class', 'Mitsubishi_Outlander Sport', 'smart_fortwo', 'Mitsubishi_Eclipse Cross', 'Chevrolet_Malibu', 'Tesla_Model Y', 'Lincoln_Aviator', 'Honda_Passport', 'Toyota_Tundra', 'Land Rover_Range Rover Sport', 'FIAT_500L', 'Nissan_Maxima', 'Chevrolet_Spark', 'Aston Martin_DB11', 'Audi_A6', 'Honda_Ridgeline', 'Buick_Cascada', 'GMC_Sierra 1500', 'Lexus_RC', 'Ford_Ecosport', 'INFINITI_QX60', 'Rolls-Royce_Dawn', 'BMW_X4', 'Toyota_Corolla', 'Chevrolet_Bolt EV', 'Aston Martin_DBS', 'Lamborghini_Aventador', 'BMW_3-Series', 'Lincoln_MKT', 'Toyota_86', 'Mazda_Mazda3 Hatchback', 'Hyundai_Elantra', 'Land Rover_Discovery Sport', 'Cadillac_XT4', 'Genesis_G80', 'McLaren_720S', 'Subaru_Ascent', 'Cadillac_XT5', 'Audi_A7', 'Bentley_Flying Spur', 'Nissan_Armada', 'Land Rover_Defender', 'Toyota_Yaris', 'Jeep_Compass', 'Nissan_Sentra', 'Ford_Taurus', 'Kia_Niro', 'FIAT_500X', 'Ford_Fusion', 'Cadillac_XT6', 'Mercedes-Benz_Metris', 'Bentley_Mulsanne', 'Hyundai_Ioniq', 'Toyota_Prius', 'Porsche_Taycan', 'McLaren_570S', 'Nissan_Altima', 'Mercedes-Benz_SLC Class', 'Volkswagen_Tiguan', 'Hyundai_NEXO', 'Kia_Soul EV', 'Ford_Super Duty F-250', 'GMC_Terrain', 'Land Rover_Range Rover Evoque', 'Nissan_NV200', 'Nissan_Pathfinder', 'BMW_7-Series', 'Honda_Clarity', 'Mercedes-Benz_GLB Class', 'Lamborghini_Huracan', 'Volvo_V60', 'Acura_TLX', 'Porsche_718 Spyder', 'Toyota_Yaris Hatchback', 'Jaguar_I-Pace', 'Rolls-Royce_Wraith', 'INFINITI_QX30', 'Alfa Romeo_4C', 'Hyundai_Kona', 'Toyota_Supra', 'BMW_X3', 'Honda_Accord', 'Tesla_Model S', 'Jeep_Cherokee', 'BMW_4-Series', 'INFINITI_QX50', 'Land Rover_Range Rover', 'Chevrolet_Cruze', 'Toyota_RAV4', 'Hyundai_Sonata', 'Ford_Transit Connect Wagon', 'Lexus_IS', 'Toyota_4Runner', 'Mercedes-Benz_GLE Class', 'Porsche_Macan', 'Lincoln_MKC', 'Audi_A8', 'Chevrolet_Tahoe', 'Kia_Optima', 'Lincoln_Continental', 'FIAT_124 Spider', 'Ferrari_Portofino', 'Mazda_MX-5 Miata', 'INFINITI_Q70', 'Kia_Rio', 'Kia_Sportage', 'Chevrolet_Camaro', 'Audi_TT', 'Chevrolet_Volt', 'Acura_MDX', 'Ford_Mustang', 'Cadillac_Escalade', 'Kia_Stinger', 'Toyota_C-HR', 'Cadillac_CTS', 'Hyundai_Palisade', 'Lexus_ES', 'Toyota_Tacoma', 'Dodge_Grand Caravan', 'Audi_R8', 'Mercedes-Benz_GLS Class', 'Jeep_Grand Cherokee', 'Toyota_Prius C', 'Subaru_Outback', 'Volkswagen_e-Golf', 'Nissan_Kicks', 'Ford_Fiesta', 'Ford_Ranger', 'GMC_Canyon', 'Chevrolet_Silverado 1500', 'Hyundai_Accent', 'BMW_5-Series', 'Ford_Explorer', 'Tesla_Model X', 'Volvo_XC90', 'Lexus_NX', 'Volvo_V90', 'INFINITI_Q50', 'BMW_6-Series', 'Jaguar_F-Type', 'MINI_Cooper', 'FIAT_500', 'Mazda_CX-3', 'GMC_Acadia', 'Chevrolet_Colorado', 'BMW_X2', 'Hyundai_Kona Electric', 'Dodge_Charger', 'Nissan_Frontier', 'Subaru_Crosstrek', 'Chevrolet_Equinox', 'Ford_F-150', 'Mercedes-Benz_SL Class', 'Buick_Lacrosse', 'Audi_A5', 'Ferrari_488 GTB', 'Volkswagen_Passat', 'ssan_370Z', 'McLaren_570GT', 'Porsche_718', 'Subaru_WRX', 'Chevrolet_Trax', 'FIAT_500e', 'Lincoln_Corsair', 'Cadillac_CT6', 'Dodge_Journey', 'Mazda_MAZDA6', 'BMW_X1', 'Mercedes-Benz_C Class', 'Mazda_CX-9', 'Lexus_LC', 'Cadillac_XTS', 'Subaru_Forester', 'Hyundai_Tucson', 'BMW_Z4', 'Rolls-Royce_Cullinan', 'Lexus_LX', 'MINI_Clubman', 'Volvo_XC60', 'Mitsubishi_Outlander', 'Alfa Romeo_4C Spider', 'Jaguar_E-Pace', 'Kia_Soul', 'Dodge_Challenger', 'Nissan_Rogue', 'Aston Martin_Vanquish', 'Jaguar_XJ', 'Jeep_Renegade']

classes1 = list(set([file.split('_')[0] + '_' + file.split('_')[1] for file in files]))

files_train, files_test = train_test_split(file_paths, test_size=0.25)
files_train, files_valid = train_test_split(files_train, test_size=0.25)

combinations_lower = [x.lower() for x in car_combinations]

ds_train = construct_ds(input_files=files_train, batch_size=32, classes=combinations_lower, input_size=(224, 224, 3), label_type='model', shuffle=True, augment=True)
ds_valid = construct_ds(input_files=files_valid, batch_size=32, classes=combinations_lower, input_size=(224, 224, 3),label_type='model', shuffle=False, augment=False)
ds_test = construct_ds(input_files=files_test, batch_size=32, classes=combinations_lower, input_size=(224, 224, 3),label_type='model', shuffle=False, augment=False)

# # Show examples from one batch
# plot_size = (18, 18)

# show_batch(ds_train, car_combinations, size=plot_size, title='Training data')
# show_batch(ds_valid, car_combinations, size=plot_size, title='Validation data')
# show_batch(ds_test, car_combinations, size=plot_size, title='Testing data')

# Init base model and compile
model = TransferModel(base='ResNet', shape=(224, 224, 3),classes=car_combinations, unfreeze='all')

model.compile(loss="categorical_crossentropy", optimizer=Adam(0.0001), metrics=["categorical_accuracy"])

class_weights = compute_class_weight(class_weight="balanced", classes=classes1, y=pd.Series([file.split('_')[0] + "_" + file.split('_')[1] for file in files]))
# class_weights = dict(zip(classes1, class_weights))
class_weights = {i:class_weights for i,class_weights in enumerate(class_weights)}

# Train model using defined tf.data.Datasets
model.history = model.train(ds_train=ds_train, ds_valid=ds_valid, epochs=10, class_weights=class_weights)

# Plot accuracy on training and validation data sets
model.plot()

# Evaluate performance on testing data
model.evaluate(ds_test=ds_test)

ds_new = construct_ds(input_files=files_train, batch_size=32, classes=combinations_lower, input_size=(224, 224, 3), label_type='model', shuffle=True, augment=True)
ds_batch = ds_new.take(1)
predictions = model.predict(ds_batch)

# OLD CODE 
# # the combination folder contains a subfolder with class and then inside class is the images
# dataset = torchvision.datasets.ImageFolder(root = './combinations')
# print(len(dataset))
# print(dataset.samples[0][0])

# val_size = int(len(dataset)*0.001)
# train_size = len(dataset)- int(len(dataset)*0.001)
# train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
# print(len(train_set))
# print(len(val_set))

# print(val_set.indices)
# print(len(val_set.indices))

# file_list=os.listdir(r'./combinations/class')
# name = file_list[val_set.indices[0]]
# brand = name.split('_')
# brand = brand[0]
# print(file_list[val_set.indices[0]])
# print(brand)
