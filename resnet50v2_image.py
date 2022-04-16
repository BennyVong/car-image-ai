import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam

from dataset_make import construct_ds, show_batch
from model import TransferModel

files = [file for file in os.listdir('./combinations/class') if file.endswith(".jpg")]
file_paths = ['./combinations/class/' + file for file in files]
car_combinations = ['Ford_Expedition', 'BMW_i8', 'Toyota_Avalon', 'Chevrolet_Blazer', 'Ford_Edge', 'Lexus_LS', 'Lamborghini_Urus', 'Acura_RLX', 'Nissan_Titan', 'Chevrolet_Traverse', 'Hyundai_Veloster', 'Ford_Flex', 'BMW_X7', 'Buick_Encore', 'Maserati_GranTurismo', 'Toyota_Camry', 'Volkswagen_Beetle', 'Chevrolet_Corvette', 'Mercedes-Benz_A Class', 'Dodge_Durango', 'Rolls-Royce_Phantom', 'Jeep_Gladiator', 'Alfa Romeo_Giulia', 'Mercedes-Benz_CLA Class', 'Mercedes-Benz_GLA Class', 'Cadillac_CT5', 'Chevrolet_Impala', 'Nissan_Leaf', 'Bentley_Continental GT', 'Audi_e-tron', 'Acura_NSX', 'Ram_1500', 'GMC_Yukon', 'Jeep_Wrangler', 'Lincoln_MKZ', 'Buick_Envision', 'Aston Martin_Vantage', 'Subaru_Impreza', 'INFINITI_Q60', 'Nissan_Rogue Sport', 'Genesis_G70', 'INFINITI_QX80', 'Mercedes-Benz_EQC', 'Lexus_RX', 'Porsche_Cayenne', 'Kia_Forte', 'Honda_HR-V', 'Nissan_GT-R', 'Subaru_BRZ', 'Kia_Telluride', 'Volkswagen_Jetta', 'Kia_Cadenza', 'Acura_RDX', 'Honda_Civic', 'Honda_CR-V', 'Land Rover_Range Rover Velar', 'Mercedes-Benz_GLC Class', 'Mitsubishi_Outlander Sport', 'smart_fortwo', 'Mitsubishi_Eclipse Cross', 'Chevrolet_Malibu', 'Tesla_Model Y', 'Lincoln_Aviator', 'Honda_Passport', 'Toyota_Tundra', 'Land Rover_Range Rover Sport', 'FIAT_500L', 'Nissan_Maxima', 'Chevrolet_Spark', 'Aston Martin_DB11', 'Audi_A6', 'Honda_Ridgeline', 'Buick_Cascada', 'GMC_Sierra 1500', 'Lexus_RC', 'Ford_Ecosport', 'INFINITI_QX60', 'Rolls-Royce_Dawn', 'BMW_X4', 'Toyota_Corolla', 'Chevrolet_Bolt EV', 'Aston Martin_DBS', 'Lamborghini_Aventador', 'BMW_3-Series', 'Lincoln_MKT', 'Toyota_86', 'Mazda_Mazda3 Hatchback', 'Hyundai_Elantra', 'Land Rover_Discovery Sport', 'Cadillac_XT4', 'Genesis_G80', 'McLaren_720S', 'Subaru_Ascent', 'Cadillac_XT5', 'Audi_A7', 'Bentley_Flying Spur', 'Nissan_Armada', 'Land Rover_Defender', 'Toyota_Yaris', 'Jeep_Compass', 'Nissan_Sentra', 'Ford_Taurus', 'Kia_Niro', 'FIAT_500X', 'Ford_Fusion', 'Cadillac_XT6', 'Mercedes-Benz_Metris', 'Bentley_Mulsanne', 'Hyundai_Ioniq', 'Toyota_Prius', 'Porsche_Taycan', 'McLaren_570S', 'Nissan_Altima', 'Mercedes-Benz_SLC Class', 'Volkswagen_Tiguan', 'Hyundai_NEXO', 'Kia_Soul EV', 'Ford_Super Duty F-250', 'GMC_Terrain', 'Land Rover_Range Rover Evoque', 'Nissan_NV200', 'Nissan_Pathfinder', 'BMW_7-Series', 'Honda_Clarity', 'Mercedes-Benz_GLB Class', 'Lamborghini_Huracan', 'Volvo_V60', 'Acura_TLX', 'Porsche_718 Spyder', 'Toyota_Yaris Hatchback', 'Jaguar_I-Pace', 'Rolls-Royce_Wraith', 'INFINITI_QX30', 'Alfa Romeo_4C', 'Hyundai_Kona', 'Toyota_Supra', 'BMW_X3', 'Honda_Accord', 'Tesla_Model S', 'Jeep_Cherokee', 'BMW_4-Series', 'INFINITI_QX50', 'Land Rover_Range Rover', 'Chevrolet_Cruze', 'Toyota_RAV4', 'Hyundai_Sonata', 'Ford_Transit Connect Wagon', 'Lexus_IS', 'Toyota_4Runner', 'Mercedes-Benz_GLE Class', 'Porsche_Macan', 'Lincoln_MKC', 'Audi_A8', 'Chevrolet_Tahoe', 'Kia_Optima', 'Lincoln_Continental', 'FIAT_124 Spider', 'Ferrari_Portofino', 'Mazda_MX-5 Miata', 'INFINITI_Q70', 'Kia_Rio', 'Kia_Sportage', 'Chevrolet_Camaro', 'Audi_TT', 'Chevrolet_Volt', 'Acura_MDX', 'Ford_Mustang', 'Cadillac_Escalade', 'Kia_Stinger', 'Toyota_C-HR', 'Cadillac_CTS', 'Hyundai_Palisade', 'Lexus_ES', 'Toyota_Tacoma', 'Dodge_Grand Caravan', 'Audi_R8', 'Mercedes-Benz_GLS Class', 'Jeep_Grand Cherokee', 'Toyota_Prius C', 'Subaru_Outback', 'Volkswagen_e-Golf', 'Nissan_Kicks', 'Ford_Fiesta', 'Ford_Ranger', 'GMC_Canyon', 'Chevrolet_Silverado 1500', 'Hyundai_Accent', 'BMW_5-Series', 'Ford_Explorer', 'Tesla_Model X', 'Volvo_XC90', 'Lexus_NX', 'Volvo_V90', 'INFINITI_Q50', 'BMW_6-Series', 'Jaguar_F-Type', 'MINI_Cooper', 'FIAT_500', 'Mazda_CX-3', 'GMC_Acadia', 'Chevrolet_Colorado', 'BMW_X2', 'Hyundai_Kona Electric', 'Dodge_Charger', 'Nissan_Frontier', 'Subaru_Crosstrek', 'Chevrolet_Equinox', 'Ford_F-150', 'Mercedes-Benz_SL Class', 'Buick_Lacrosse', 'Audi_A5', 'Ferrari_488 GTB', 'Volkswagen_Passat', 'ssan_370Z', 'McLaren_570GT', 'Porsche_718', 'Subaru_WRX', 'Chevrolet_Trax', 'FIAT_500e', 'Lincoln_Corsair', 'Cadillac_CT6', 'Dodge_Journey', 'Mazda_MAZDA6', 'BMW_X1', 'Mercedes-Benz_C Class', 'Mazda_CX-9', 'Lexus_LC', 'Cadillac_XTS', 'Subaru_Forester', 'Hyundai_Tucson', 'BMW_Z4', 'Rolls-Royce_Cullinan', 'Lexus_LX', 'MINI_Clubman', 'Volvo_XC60', 'Mitsubishi_Outlander', 'Alfa Romeo_4C Spider', 'Jaguar_E-Pace', 'Kia_Soul', 'Dodge_Challenger', 'Nissan_Rogue', 'Aston Martin_Vanquish', 'Jaguar_XJ', 'Jeep_Renegade']

classes1 = list(set([file.split('_')[0] + '_' + file.split('_')[1] for file in files]))

files_train, files_test = train_test_split(file_paths, test_size=0.25)
files_train, files_valid = train_test_split(files_train, test_size=0.25)

combinations_lower = [x.lower() for x in car_combinations]

ds_train = construct_ds(input_files=files_train, batch_size=32, classes=combinations_lower, input_size=(224, 224, 3), label_type='model', shuffle=True, augment=True)
ds_valid = construct_ds(input_files=files_valid, batch_size=32, classes=combinations_lower, input_size=(224, 224, 3),label_type='model', shuffle=False, augment=False)
ds_test = construct_ds(input_files=files_test, batch_size=32, classes=combinations_lower, input_size=(224, 224, 3),label_type='model', shuffle=False, augment=False)

# plot_size = (18, 18)

# show_batch(ds_train, car_combinations, size=plot_size, title='Training data')
# show_batch(ds_valid, car_combinations, size=plot_size, title='Validation data')
# show_batch(ds_test, car_combinations, size=plot_size, title='Testing data')

model = TransferModel(base='ResNet', shape=(224, 224, 3),classes=car_combinations, unfreeze='all')

model.compile(loss="categorical_crossentropy", optimizer=Adam(0.0001), metrics=["categorical_accuracy"])

class_weights = compute_class_weight(class_weight="balanced", classes=classes1, y=pd.Series([file.split('_')[0] + "_" + file.split('_')[1] for file in files]))
# class_weights = dict(zip(classes1, class_weights))
class_weights = {i:class_weights for i,class_weights in enumerate(class_weights)}

model.history = model.train(ds_train=ds_train, ds_valid=ds_valid, epochs=10, class_weights=class_weights)

model.plot()

model.evaluate(ds_test=ds_test)

ds_new = construct_ds(input_files=files_train, batch_size=32, classes=combinations_lower, input_size=(224, 224, 3), label_type='model', shuffle=True, augment=True)
ds_batch = ds_new.take(1)
predictions = model.predict(ds_batch)