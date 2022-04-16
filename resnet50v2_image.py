import os
import sys

import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from dataset_make import construct_ds, show_batch
from model import TransferModel

mode = sys.argv[1]
model_label = sys.argv[2]

files = [file for file in os.listdir('./combinations/class') if file.endswith(".jpg")]
file_paths = ['./combinations/class/' + file for file in files]
car_combinations = ['Mercedes-Benz_GLC Class', 'Ford_Edge', 'Maserati_Ghibli', 'Mercedes-Benz_E Class', 'GMC_Yukon', 'Honda_Ridgeline', 'Porsche_911', 'Ferrari_488 GTB', 'Lexus_LS', 'Dodge_Grand Caravan', 'Mercedes-Benz_EQC', 'Subaru_Crosstrek', 'Kia_Sportage', 'Hyundai_Ioniq', 'Hyundai_Kona Electric', 'Toyota_4Runner', 'Acura_TLX', 'Land Rover_Discovery Sport', 'Honda_Civic', 'FIAT_500e', 'Mercedes-Benz_G Class', 'Volvo_XC40', 'Honda_Odyssey', 'Honda_Insight', 'Ford_Fusion', 'Nissan_GT-R', 'Toyota_Tundra', 'BMW_2-Series', 'Lincoln_Nautilus', 'Volvo_S60', 'Kia_Telluride', 'Aston Martin_Vanquish', 'Nissan_Altima', 'Chevrolet_Traverse', 'Tesla_Model X', 'Nissan_Maxima', 'Subaru_Outback', 'BMW_5-Series', 'Mercedes-Benz_SL Class', 'Bentley_Mulsanne', 'Alfa Romeo_4C', 'Lincoln_Corsair', 'Audi_R8', 'BMW_6-Series', 'Porsche_Taycan', 'Jaguar_XJ', 'Ferrari_GTC4Lusso', 'Chevrolet_Volt', 'Chevrolet_Tahoe', 'Chrysler_300', 'Ford_Expedition', 'Rolls-Royce_Wraith', 'Chevrolet_Spark', 'Subaru_STI S209', 'Lincoln_Navigator', 'Jaguar_E-Pace', 'Volkswagen_Beetle', 'Audi_Q8', 'Lexus_IS', 'BMW_i8', 'Volvo_XC90', 'Ford_Flex', 'Acura_ILX', 'Nissan_Rogue Sport', 'Cadillac_XT4', 'Chevrolet_Colorado', 'Jaguar_I-Pace', 'INFINITI_Q60', 'Mazda_CX-30', 'Hyundai_Kona', 'Toyota_Avalon', 'Maserati_Levante', 'BMW_X6', 'Nissan_Titan', 'Porsche_718 Spyder', 'Ford_Escape', 'Nissan_Kicks', 'Chevrolet_Camaro', 'Subaru_WRX', 'Buick_Regal', 'Audi_A5', 'BMW_7-Series', 'Hyundai_Santa Fe', 'Bentley_Bentayga', 'INFINITI_Q70', 'McLaren_570S', 'Cadillac_Escalade', 'Cadillac_XT6', 'Cadillac_XT5', 'Chevrolet_Silverado 1500', 'Ram_1500', 'Ford_F-150', 'Lexus_NX', 'Genesis_G80', 'Mercedes-Benz_GLA Class', 'Volkswagen_Passat', 'Land Rover_Range Rover Sport',
                    'Nissan_Leaf', 'Honda_Pilot', 'Volkswagen_Atlas', 'Lincoln_MKC', 'Mercedes-Benz_S Class', 'Kia_K900', 'Toyota_Sequoia', 'Hyundai_Accent', 'Mercedes-Benz_C Class', 'Porsche_Panamera', 'Volkswagen_Golf', 'Chevrolet_Equinox', 'Chevrolet_TrailBlazer', 'Volvo_XC60', 'Audi_A7', 'Lexus_GS', 'Dodge_Charger', 'Acura_RLX', 'Honda_CR-V', 'Land Rover_Range Rover Velar', 'Dodge_Journey', 'BMW_i3', 'FIAT_500', 'Lamborghini_Aventador', 'Lincoln_MKT', 'Genesis_G70', 'Audi_A6', 'Bentley_Flying Spur', 'Hyundai_Tucson', 'Ford_Transit Connect Wagon', 'McLaren_720S', 'Ford_Ecosport', 'Alfa Romeo_Stelvio', 'Jaguar_XF', 'Porsche_Cayenne', 'Lamborghini_Urus', 'Lexus_RC', 'BMW_4-Series', 'INFINITI_QX60', 'Acura_RDX', 'MINI_Cooper', 'Chevrolet_Blazer', 'Jaguar_F-Pace', 'Mazda_CX-5', 'Audi_Q7', 'Subaru_Impreza', 'Volkswagen_Tiguan', 'Land Rover_Discovery', 'Kia_Soul', 'Toyota_RAV4', 'Volvo_V60', 'Porsche_Macan', 'Land Rover_Range Rover', 'Mercedes-Benz_GLE Class', 'BMW_X3', 'FIAT_124 Spider', 'Subaru_BRZ', 'Toyota_Sienna', 'Jeep_Grand Cherokee', 'Dodge_Durango', 'Kia_Sedona', 'Honda_Passport', 'Rolls-Royce_Ghost', 'Volkswagen_Jetta', 'Hyundai_Sonata', 'Acura_NSX', 'Cadillac_CT5', 'Toyota_Highlander', 'Tesla_Model S', 'Hyundai_NEXO', 'Bentley_Continental GT', 'Ford_Mustang', 'Lexus_RX', 'Chevrolet_Sonic', 'Chevrolet_Trax', 'Aston Martin_DB11', 'Rolls-Royce_Phantom', 'Nissan_Pathfinder', 'Ford_Explorer', 'BMW_X4', 'Mercedes-Benz_Metris', 'INFINITI_QX80', 'Lincoln_MKZ', 'Maserati_GranTurismo', 'Mazda_CX-3', 'FIAT_500L', 'Mitsubishi_Outlander Sport', 'Hyundai_Veloster', 'Land Rover_Defender', 'Genesis_G90', 'Toyota_Yaris', 'Chevrolet_Suburban', 'Nissan_370Z', 'BMW_X5', 'Ram_2500', 'INFINITI_Q50', 'BMW_X2', 'Mercedes-Benz_SLC Class', 'Lexus_GX', 'Mercedes-Benz_CLS Class', 'Audi_A4', 'Ford_Taurus', 'Toyota_Land Cruiser', 'Kia_Sorento', 'Lexus_LC', 'BMW_Z4', 'GMC_Sierra 2500HD', 'Jeep_Gladiator', 'BMW_X7', 'Chevrolet_Impala', 'Rolls-Royce_Cullinan', 'Lexus_UX', 'Rolls-Royce_Dawn', 'Hyundai_Palisade', 'Chrysler_Pacifica', 'Mazda_CX-9', 'Toyota_C-HR', 'Lamborghini_Huracan', 'Buick_Enclave', 'Mazda_MAZDA6', 'GMC_Acadia', 'Subaru_Legacy', 'Aston Martin_Vantage', 'Mitsubishi_Outlander', 'Kia_Optima', 'MINI_Cooper Countryman', 'Kia_Niro', 'smart_fortwo', 'Alfa Romeo_4C Spider', 'Subaru_Ascent', 'Chevrolet_Bolt EV', 'Jeep_Cherokee', 'Buick_Envision', 'Kia_Rio', 'GMC_Sierra 1500', 'Toyota_Supra', 'Nissan_Rogue', 'Kia_Forte', 'Honda_HR-V', 'Lexus_LX', 'BMW_3-Series', 'GMC_Canyon',
                    'Chevrolet_Malibu', 'Audi_Q3', 'Maserati_Quattroporte', 'Hyundai_Elantra', 'FIAT_500X', 'BMW_X1', 'Jeep_Compass', 'Subaru_Forester', 'Chevrolet_Silverado 2500HD', 'Acura_MDX', 'GMC_Terrain',
                    'INFINITI_QX30', 'BMW_8-Series', 'Cadillac_ATS', 'Ford_Super Duty F-250', 'INFINITI_QX50', 'Mazda_MAZDA3', 'Nissan_Versa', 'Honda_Accord', 'Cadillac_XTS', 'Mitsubishi_Mirage', 'Lincoln_Aviator', 'Alfa Romeo_Giulia', 'Honda_Fit', 'Ford_Ranger', 'Kia_Cadenza', 'Tesla_Model Y', 'Audi_A3', 'Honda_Clarity', 'Porsche_718', 'Chevrolet_Cruze', 'Hyundai_Venue', 'Mazda_Mazda3 Hatchback', 'Chevrolet_Corvette', 'Ford_Fiesta', 'Toyota_86', 'MINI_Clubman', 'Jaguar_F-Type', 'Cadillac_CT6', 'Nissan_Frontier', 'Mazda_MX-5 Miata', 'Toyota_Corolla', 'Jeep_Wrangler', 'Tesla_Model 3', 'Buick_Lacrosse', 'Toyota_Tacoma', 'Ferrari_Portofino', 'Toyota_Prius', 'Audi_A8', 'Mercedes-Benz_GLS Class', 'Nissan_Murano', 'Toyota_Mirai', 'Kia_Stinger', 'Jaguar_XE', 'Mitsubishi_Eclipse Cross', 'Cadillac_CT4', 'Mercedes-Benz_A Class', 'Mercedes-Benz_GLB Class', 'Land Rover_Range Rover Evoque', 'Lincoln_Continental', 'Nissan_NV200', 'Toyota_Prius C', 'Volkswagen_Arteon', 'Jeep_Renegade', 'Lexus_ES', 'Audi_e-tron', 'Audi_Q5', 'Audi_TT', 'Nissan_Sentra', 'Dodge_Challenger', 'Cadillac_CTS', 'Aston Martin_DBS', 'Mercedes-Benz_AMG GT', 'Nissan_Armada', 'Volvo_V90', 'Volkswagen_e-Golf', 'Toyota_Camry', 'Toyota_Yaris Hatchback', 'Volvo_S90', 'Buick_Cascada', 'Kia_Soul EV', 'Buick_Encore', 'Mercedes-Benz_CLA Class', 'McLaren_570GT']

classes1 = list(set([file.split('_')[0] + '_' + file.split('_')[1] for file in files]))

files_train, files_test = train_test_split(file_paths, test_size=0.25, random_state=1337)
files_train, files_valid = train_test_split(files_train, test_size=0.25, random_state=1337)

combinations_lower = [x.lower() for x in car_combinations]

ds_train = construct_ds(input_files=files_train, batch_size=32, classes=combinations_lower, input_size=(224, 224, 3), label_type='makemodel', shuffle=True, augment=True)
ds_valid = construct_ds(input_files=files_valid, batch_size=32, classes=combinations_lower, input_size=(224, 224, 3),label_type='makemodel', shuffle=False, augment=False)
ds_test = construct_ds(input_files=files_test, batch_size=32, classes=combinations_lower, input_size=(224, 224, 3),label_type='makemodel', shuffle=False, augment=False)

# plot_size = (18, 18)

# show_batch(ds_train, car_combinations, size=plot_size, title='Training data')
# show_batch(ds_valid, car_combinations, size=plot_size, title='Validation data')
# show_batch(ds_test, car_combinations, size=plot_size, title='Testing data')

if mode == "train":
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
    os.add_dll_directory("C:/Program Files/zlib123dllx64/dll_x64")
    sys.setrecursionlimit(10000)

    model = TransferModel(base='VGG16', shape=(224, 224, 3),classes=car_combinations, unfreeze='all')

    model.compile(loss="categorical_crossentropy", optimizer=Adam(0.0001), metrics=["categorical_accuracy"])

    class_weights = compute_class_weight(class_weight="balanced", classes=classes1, y=pd.Series([file.split('_')[0] + "_" + file.split('_')[1] for file in files]))
    # class_weights = dict(zip(classes1, class_weights))
    class_weights = {i:class_weights for i,class_weights in enumerate(class_weights)}

    model.history = model.train(ds_train=ds_train, ds_valid=ds_valid, epochs=40, class_weights=class_weights, model_label=model_label)

    model.save(model_label)

elif mode == "test": 
    model = TransferModel(base='ResNet', shape=(224, 224, 3),classes=car_combinations, unfreeze='all')

    model.load("./"+model_label+"/")

    model.evaluate(ds_test)
    exit()

elif mode == "load_checkpoint":  
    model = load_model("checkpoints/" + model_label)
    
    ds_new = construct_ds(input_files=files_train, batch_size=32, classes=combinations_lower, input_size=(224, 224, 3), label_type='model', shuffle=True, augment=True)
    ds_batch = ds_new.take(1)
    predictions = model.predict(ds_batch)
    plot_size = (18, 18)
    show_batch(ds_batch, car_combinations, size=plot_size, title='Test')
    
    # prediction = max(predictions[0])
    
    predicted_class = numpy.argmax(predictions, axis=-1)
    print(predicted_class)
    exit()
    
elif mode == "test_single_image":
    model = TransferModel(base='ResNet', shape=(224, 224, 3),classes=car_combinations, unfreeze='all')

    model.load("./"+model_label+"/")

    ds_new = construct_ds(input_files=files_train, batch_size=1, classes=combinations_lower, input_size=(224, 224, 3), label_type='makemodel', shuffle=True, augment=False)
    
    ds_batch = ds_new.take(1)
    # print(ds_batch)
    
    # for images, labels in ds_test.take(1):  # only take first element of dataset
    #     numpy_images = images.numpy()
    #     numpy_labels = labels.numpy()
    # print(numpy_labels[0])
    
    predictions = model.predict(ds_batch)
    
    max = max(predictions[0])
    highest = numpy.where(predictions == max)
    print(highest[1])
    print(int(highest[1]))
    print(car_combinations[int(highest[1])])
    prediction = car_combinations[int(highest[1])]
    plot_size = (18, 18)
    show_batch(ds_batch, car_combinations, size=plot_size, title=prediction)
    
    exit()

model.plot()

model.evaluate(ds_test=ds_test)

ds_new = construct_ds(input_files=files_train, batch_size=32, classes=combinations_lower, input_size=(224, 224, 3), label_type='makemodel', shuffle=True, augment=True)
ds_batch = ds_new.take(1)
predictions = model.predict(ds_batch)
