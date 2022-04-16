import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import pickle
import numpy
import sys

def import_data():
    car_combinations = ['Mercedes-Benz_GLC Class', 'Ford_Edge', 'Maserati_Ghibli', 'Mercedes-Benz_E Class', 'GMC_Yukon', 'Honda_Ridgeline', 'Porsche_911', 'Ferrari_488 GTB', 'Lexus_LS', 'Dodge_Grand Caravan', 'Mercedes-Benz_EQC', 'Subaru_Crosstrek', 'Kia_Sportage', 'Hyundai_Ioniq', 'Hyundai_Kona Electric', 'Toyota_4Runner', 'Acura_TLX', 'Land Rover_Discovery Sport', 'Honda_Civic', 'FIAT_500e', 'Mercedes-Benz_G Class', 'Volvo_XC40', 'Honda_Odyssey', 'Honda_Insight', 'Ford_Fusion', 'Nissan_GT-R', 'Toyota_Tundra', 'BMW_2-Series', 'Lincoln_Nautilus', 'Volvo_S60', 'Kia_Telluride', 'Aston Martin_Vanquish', 'Nissan_Altima', 'Chevrolet_Traverse', 'Tesla_Model X', 'Nissan_Maxima', 'Subaru_Outback', 'BMW_5-Series', 'Mercedes-Benz_SL Class', 'Bentley_Mulsanne', 'Alfa Romeo_4C', 'Lincoln_Corsair', 'Audi_R8', 'BMW_6-Series', 'Porsche_Taycan', 'Jaguar_XJ', 'Ferrari_GTC4Lusso', 'Chevrolet_Volt', 'Chevrolet_Tahoe', 'Chrysler_300', 'Ford_Expedition', 'Rolls-Royce_Wraith', 'Chevrolet_Spark', 'Subaru_STI S209', 'Lincoln_Navigator', 'Jaguar_E-Pace', 'Volkswagen_Beetle', 'Audi_Q8', 'Lexus_IS', 'BMW_i8', 'Volvo_XC90', 'Ford_Flex', 'Acura_ILX', 'Nissan_Rogue Sport', 'Cadillac_XT4', 'Chevrolet_Colorado', 'Jaguar_I-Pace', 'INFINITI_Q60', 'Mazda_CX-30', 'Hyundai_Kona', 'Toyota_Avalon', 'Maserati_Levante', 'BMW_X6', 'Nissan_Titan', 'Porsche_718 Spyder', 'Ford_Escape', 'Nissan_Kicks', 'Chevrolet_Camaro', 'Subaru_WRX', 'Buick_Regal', 'Audi_A5', 'BMW_7-Series', 'Hyundai_Santa Fe', 'Bentley_Bentayga', 'INFINITI_Q70', 'McLaren_570S', 'Cadillac_Escalade', 'Cadillac_XT6', 'Cadillac_XT5', 'Chevrolet_Silverado 1500', 'Ram_1500', 'Ford_F-150', 'Lexus_NX', 'Genesis_G80', 'Mercedes-Benz_GLA Class', 'Volkswagen_Passat', 'Land Rover_Range Rover Sport',
                        'Nissan_Leaf', 'Honda_Pilot', 'Volkswagen_Atlas', 'Lincoln_MKC', 'Mercedes-Benz_S Class', 'Kia_K900', 'Toyota_Sequoia', 'Hyundai_Accent', 'Mercedes-Benz_C Class', 'Porsche_Panamera', 'Volkswagen_Golf', 'Chevrolet_Equinox', 'Chevrolet_TrailBlazer', 'Volvo_XC60', 'Audi_A7', 'Lexus_GS', 'Dodge_Charger', 'Acura_RLX', 'Honda_CR-V', 'Land Rover_Range Rover Velar', 'Dodge_Journey', 'BMW_i3', 'FIAT_500', 'Lamborghini_Aventador', 'Lincoln_MKT', 'Genesis_G70', 'Audi_A6', 'Bentley_Flying Spur', 'Hyundai_Tucson', 'Ford_Transit Connect Wagon', 'McLaren_720S', 'Ford_Ecosport', 'Alfa Romeo_Stelvio', 'Jaguar_XF', 'Porsche_Cayenne', 'Lamborghini_Urus', 'Lexus_RC', 'BMW_4-Series', 'INFINITI_QX60', 'Acura_RDX', 'MINI_Cooper', 'Chevrolet_Blazer', 'Jaguar_F-Pace', 'Mazda_CX-5', 'Audi_Q7', 'Subaru_Impreza', 'Volkswagen_Tiguan', 'Land Rover_Discovery', 'Kia_Soul', 'Toyota_RAV4', 'Volvo_V60', 'Porsche_Macan', 'Land Rover_Range Rover', 'Mercedes-Benz_GLE Class', 'BMW_X3', 'FIAT_124 Spider', 'Subaru_BRZ', 'Toyota_Sienna', 'Jeep_Grand Cherokee', 'Dodge_Durango', 'Kia_Sedona', 'Honda_Passport', 'Rolls-Royce_Ghost', 'Volkswagen_Jetta', 'Hyundai_Sonata', 'Acura_NSX', 'Cadillac_CT5', 'Toyota_Highlander', 'Tesla_Model S', 'Hyundai_NEXO', 'Bentley_Continental GT', 'Ford_Mustang', 'Lexus_RX', 'Chevrolet_Sonic', 'Chevrolet_Trax', 'Aston Martin_DB11', 'Rolls-Royce_Phantom', 'Nissan_Pathfinder', 'Ford_Explorer', 'BMW_X4', 'Mercedes-Benz_Metris', 'INFINITI_QX80', 'Lincoln_MKZ', 'Maserati_GranTurismo', 'Mazda_CX-3', 'FIAT_500L', 'Mitsubishi_Outlander Sport', 'Hyundai_Veloster', 'Land Rover_Defender', 'Genesis_G90', 'Toyota_Yaris', 'Chevrolet_Suburban', 'Nissan_370Z', 'BMW_X5', 'Ram_2500', 'INFINITI_Q50', 'BMW_X2', 'Mercedes-Benz_SLC Class', 'Lexus_GX', 'Mercedes-Benz_CLS Class', 'Audi_A4', 'Ford_Taurus', 'Toyota_Land Cruiser', 'Kia_Sorento', 'Lexus_LC', 'BMW_Z4', 'GMC_Sierra 2500HD', 'Jeep_Gladiator', 'BMW_X7', 'Chevrolet_Impala', 'Rolls-Royce_Cullinan', 'Lexus_UX', 'Rolls-Royce_Dawn', 'Hyundai_Palisade', 'Chrysler_Pacifica', 'Mazda_CX-9', 'Toyota_C-HR', 'Lamborghini_Huracan', 'Buick_Enclave', 'Mazda_MAZDA6', 'GMC_Acadia', 'Subaru_Legacy', 'Aston Martin_Vantage', 'Mitsubishi_Outlander', 'Kia_Optima', 'MINI_Cooper Countryman', 'Kia_Niro', 'smart_fortwo', 'Alfa Romeo_4C Spider', 'Subaru_Ascent', 'Chevrolet_Bolt EV', 'Jeep_Cherokee', 'Buick_Envision', 'Kia_Rio', 'GMC_Sierra 1500', 'Toyota_Supra', 'Nissan_Rogue', 'Kia_Forte', 'Honda_HR-V', 'Lexus_LX', 'BMW_3-Series', 'GMC_Canyon',
                        'Chevrolet_Malibu', 'Audi_Q3', 'Maserati_Quattroporte', 'Hyundai_Elantra', 'FIAT_500X', 'BMW_X1', 'Jeep_Compass', 'Subaru_Forester', 'Chevrolet_Silverado 2500HD', 'Acura_MDX', 'GMC_Terrain',
                        'INFINITI_QX30', 'BMW_8-Series', 'Cadillac_ATS', 'Ford_Super Duty F-250', 'INFINITI_QX50', 'Mazda_MAZDA3', 'Nissan_Versa', 'Honda_Accord', 'Cadillac_XTS', 'Mitsubishi_Mirage', 'Lincoln_Aviator', 'Alfa Romeo_Giulia', 'Honda_Fit', 'Ford_Ranger', 'Kia_Cadenza', 'Tesla_Model Y', 'Audi_A3', 'Honda_Clarity', 'Porsche_718', 'Chevrolet_Cruze', 'Hyundai_Venue', 'Mazda_Mazda3 Hatchback', 'Chevrolet_Corvette', 'Ford_Fiesta', 'Toyota_86', 'MINI_Clubman', 'Jaguar_F-Type', 'Cadillac_CT6', 'Nissan_Frontier', 'Mazda_MX-5 Miata', 'Toyota_Corolla', 'Jeep_Wrangler', 'Tesla_Model 3', 'Buick_Lacrosse', 'Toyota_Tacoma', 'Ferrari_Portofino', 'Toyota_Prius', 'Audi_A8', 'Mercedes-Benz_GLS Class', 'Nissan_Murano', 'Toyota_Mirai', 'Kia_Stinger', 'Jaguar_XE', 'Mitsubishi_Eclipse Cross', 'Cadillac_CT4', 'Mercedes-Benz_A Class', 'Mercedes-Benz_GLB Class', 'Land Rover_Range Rover Evoque', 'Lincoln_Continental', 'Nissan_NV200', 'Toyota_Prius C', 'Volkswagen_Arteon', 'Jeep_Renegade', 'Lexus_ES', 'Audi_e-tron', 'Audi_Q5', 'Audi_TT', 'Nissan_Sentra', 'Dodge_Challenger', 'Cadillac_CTS', 'Aston Martin_DBS', 'Mercedes-Benz_AMG GT', 'Nissan_Armada', 'Volvo_V90', 'Volkswagen_e-Golf', 'Toyota_Camry', 'Toyota_Yaris Hatchback', 'Volvo_S90', 'Buick_Cascada', 'Kia_Soul EV', 'Buick_Encore', 'Mercedes-Benz_CLA Class', 'McLaren_570GT']

    files = [file for file in os.listdir('./combinations/class') if file.endswith(".jpg")]
    entries = [file for file in files]
    
    entries = list([list([file.split('_')[0] + "_" + file.split('_')[1], file.split('_')[3]]) for file in entries])

    for entry in list(entries):
        if 'nan' in entry[0] or 'nan' in entry[1]:
            entries.remove(entry)

    x, y = list([str(entry[0]) for entry in entries]), list([int(entry[1]) for entry in entries])

    x = list([int(car_combinations.index(model)) for model in x])

    x = numpy.array(x)
    y = numpy.array(y)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=1337)
    xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, test_size=0.25, random_state=1337)

    return xtrain, xtest, ytrain, ytest, xvalid, yvalid


@ignore_warnings(category=ConvergenceWarning)
def train_decision_tree():
    xtrain, xtest, ytrain, ytest, xvalid, yvalid = import_data()
    from sklearn.ensemble import RandomForestClassifier
    pipeline = Pipeline([
        ('classification', RandomForestClassifier())
    ])
    pipeline.fit(xtrain.reshape(-1, 1), ytrain)
    return pipeline, xtest, ytest


def train(model_name):
    model, xtest, ytest = train_decision_tree()
    file = open(model_name + "_label.model", "wb")
    pickle.dump(model, file)
    file.close
    yprediction = model.predict(xtest.reshape(-1, 1))
    actual_label = ytest

    i = 0
    differences = []
    perfect = 0
    counts = {}
    for prediction in yprediction:
        difference = prediction / actual_label[i]
        differences.append(difference)
        if difference == 1:
            perfect += 1
        i += 1
        if difference in counts:
            counts[difference] += 1
        else:
            counts[difference] = 1

    print("average difference:", sum(differences)/len(differences))
    print("highest overestimate:", max(differences))
    print("lowest underestimate:", min(differences))
    print("perfect estimates", perfect, "/", len(differences))
    

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    differences = list(counts.keys())
    counts_graphed = list(counts.values())
    ax.set_ylabel('Number of Predictions')
    ax.set_xlabel('Prediction/Actual Ratio')

    ax.set_title('Number of Predictions for each Ratio')
    plt.bar(differences, height=counts_graphed, width=0.01)
    plt.show()
    
def predict_price(model_name, car_predictions):
    file = open(model_name + ".model", 'rb')
    model = pickle.load(file)
    file.close
    predictions = model.predict(car_predictions.reshape(-1, 1))
    return predictions

def main():
    mode = sys.argv[1]
    model_name = sys.argv[2]
    
    if mode == "training":
        train(model_name)
    
if __name__ == "__main__":
    main()
