from unittest import loader
from sklearn import datasets
import torch
import torchvision
import os
from torch.utils.data import DataLoader
from PIL import Image

car_brands = ["Acura", "Alfa Romeo", "Aston Martin", "Audi", "Bentley", "BMW", "Buick", "Cadillac", "Chevrolet", "Chrysler", "Dodge", "Ferrari", "FIAT", "Ford", "Genesis", "GMC", "Honda", "Hyundai", "INFINITI", "Jaguar", "Jeep", "Kia", "Lamborghini", "Land Rover", "Lexus", "Lincoln", "Maserati", "Mazda", "McLaren", "Mercedes-Benz", "MINI", "Mitsubishi", "Nissan", "Porsche", "Ram", "Rolls-Royce", "smart", "Subaru", "Tesla", "Toyota", "Volkswagen", "Volvo"]

car_makes = ["ILX", "MDX", "NSX", "RDX", "RLX", "TLX", "4C Spider", "4C", "Giulia", "Stelvio", "DB11", "DBS", "Vanquish", "Vantage", "A3", "A4", "A5", "A6", "A7", "A8", "e-tron", "Q3", "Q5", "Q7", "Q8", "R8", "TT", "Bentayga", "Continental GT", "Flying Spur", "Mulsanne", "2-Series", "3-Series", "4-Series", "5-Series", "6-Series", "7-Series", "8-Series", "i3", "i8", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "Z4", "Cascada", "Enclave", "Encore", "Envision", "Lacrosse", "Regal", "ATS", "CT4", "CT5", "CT6", "CTS", "Escalade", "XT4", "XT5", "XT6", "XTS", "Blazer", "Bolt EV", "Camaro", "Colorado", "Corvette", "Cruze", "Equinox", "Impala", "Malibu", "Silverado 1500", "Silverado 2500HD", "Sonic", "Spark", "Suburban", "Tahoe", "TrailBlazer", "Traverse", "Trax", "Volt", "300", "Pacifica", "Challenger", "Charger", "Durango", "Grand Caravan", "Journey", "488 GTB", "GTC4Lusso", "Portofino", "124 Spider", "500e", "500L", "500X", "500", "Ecosport", "Edge", "Escape", "Expedition", "Explorer", "F-150", "Fiesta", "Flex", "Fusion", "Mustang", "Ranger", "Super Duty F-250", "Taurus", "Transit Connect Wagon", "G70", "G80", "G90", "Acadia", "Canyon", "Sierra 1500", "Sierra 2500HD", "Terrain", "Yukon", "Accord", "Civic", "Clarity", "CR-V", "Fit", "HR-V", "Insight", "Odyssey", "Passport", "Pilot", "Ridgeline", "Accent", "Elantra", "Ioniq", "Kona Electric", "Kona", "NEXO", "Palisade", "Santa Fe", "Sonata", "Tucson", "Veloster", "Venue", "Q50", "Q60", "Q70", "QX30", "QX50", "QX60", "QX80", "E-Pace", "F-Pace", "F-Type", "I-Pace", "XE", "XF", "XJ", "Cherokee", "Compass", "Gladiator", "Grand Cherokee", "Renegade", "Wrangler", "Cadenza", "Forte", "K900", "Niro", "Optima", "Rio", "Sedona", "Sorento", "Soul EV", "Soul", "Sportage", "Stinger", "Telluride", "Aventador", "Huracan", "Urus", "Defender", "Discovery Sport", "Discovery", "Range Rover Evoque", "Range Rover Sport", "Range Rover Velar", "Range Rover", "ES", "GS", "GX", "IS", "LC", "LS", "LX", "NX", "RC", "RX", "UX", "Aviator", "Continental", "Corsair", "MKC", "MKT", "MKZ", "Nautilus", "Navigator", "Ghibli", "GranTurismo", "Levante", "Quattroporte", "CX-30", "CX-3", "CX-5", "CX-9", "Mazda3 Hatchback", "MAZDA3", "MAZDA6", "MX-5 Miata", "570GT", "570S", "720S", "A Class", "AMG GT", "C Class", "CLA Class", "CLS Class", "E Class", "EQC", "G Class", "GLA Class", "GLB Class", "GLC Class", "GLE Class", "GLS Class", "Metris", "S Class", "SL Class", "SLC Class", "Clubman", "Cooper Countryman", "Cooper", "Eclipse Cross", "Mirage", "Outlander Sport", "Outlander", "370Z", "Altima", "Armada", "Frontier", "GT-R", "Kicks", "Leaf", "Maxima", "Murano", "NV200", "Pathfinder", "Rogue Sport", "Rogue", "Sentra", "Titan", "Versa", "718 Spyder", "718", "911", "Cayenne", "Macan", "Panamera", "Taycan", "1500", "2500", "Cullinan", "Dawn", "Ghost", "Phantom", "Wraith", "fortwo", "Ascent", "BRZ", "Crosstrek", "Forester", "Impreza", "Legacy", "Outback", "STI S209", "WRX", "Model 3", "Model S", "Model X", "Model Y", "4Runner", "86", "Avalon", "C-HR", "Camry", "Corolla", "Highlander", "Land Cruiser", "Mirai", "Prius C", "Prius", "RAV4", "Sequoia", "Sienna", "Supra", "Tacoma", "Tundra", "Yaris Hatchback", "Yaris", "Arteon", "Atlas"]

# the combination folder contains a subfolder with class and then inside class is the images
dataset = torchvision.datasets.ImageFolder(root = './combinations')
print(len(dataset))
print(dataset.samples[0][0])

val_size = int(len(dataset)*0.001)
train_size = len(dataset)- int(len(dataset)*0.001)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
print(len(train_set))
print(len(val_set))

print(val_set.indices)
print(len(val_set.indices))

file_list=os.listdir(r'./combinations/class')
name = file_list[val_set.indices[0]]
brand = name.split('_')
brand = brand[0]
print(file_list[val_set.indices[0]])
print(brand)