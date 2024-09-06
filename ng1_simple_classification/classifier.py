# Add more imports as/if needed...
from data import class_type, data_train, features
from functools import cache
# Allowed imports: Modules included in Python (see https://docs.python.org/3.10/py-modindex.html),
# and external libraries listed in requirements.txt


# Helper function for showing data (not part of assignment)
def print_data():
    """ Print animal name and feature names and values """
    for animal_name, animal_features in data_train.items():
        feature_string = ', '.join([f'{feature_name} = {feature_value}'
                                    for feature_name, feature_value in zip(features, animal_features)])
        print(f'{animal_name}: ' + feature_string)


# Modify the code below to return "sensible" classes based on input features.
# Feel free to add and remove code, the silly if statements below are only
# meant to show how features can be used and combined.
# The code returns the placeholder integer 0 in every case. The final classifier
# must return integers in the range 1-7 (corresponding to 7 animal classes)

# Store the result so the averages don't have to be calculated every time an animal gets classified
@cache  
def get_animal_averages() -> list:
    '''Returns a list containing the "average" features of an animal for each of the animal classes'''
    animal_amounts = [0, 0, 0, 0, 0, 0, 0]
    animal_features_averages = []
    for i in range(7):
        animal_features_averages.append([0.0] * 16)

    # Sum the features
    for values in data_train.values():
        animal_type = values[-1] - 1
        animal_amounts[animal_type] += 1
        for i, v in enumerate(values[0 : -1]):
            animal_features_averages[animal_type][i] += v 

    # Divide by the amount of animals from each class to get the average
    for i, amount in enumerate(animal_amounts):
        animal_stats = animal_features_averages[i]
        for j in range(len(animal_stats)):
            animal_stats[j] /= amount

    return animal_features_averages

            
def classify_animal(hair, feathers, eggs, milk, airborne, aquatic, predator, toothed,
                    backbone, breathes, venomous, fins, legs, tail, domestic, catsize):
    """ Classifies animal based on 16 features 

    Arguments:
    ----------
    hair, feathers, ... : int
        Features descibing an animal

    Returns:
    ---------
    class_int: int
        Integer in range 1-7 corresponding to 7 classes of animal

    """
    ANIMAL_CLASSES = 7
    animal_features_averages = get_animal_averages()
    features = [hair, feathers, eggs, milk,
                airborne, aquatic, predator, toothed,
                backbone, breathes, venomous, fins,
                legs, tail, domestic, catsize]
    
    # counts up how much the input animal is different from each of the averages.
    # the smallest value will be used to classify the input animal.
    different_from_average = [0.0] * ANIMAL_CLASSES
    for i in range(ANIMAL_CLASSES):
        averages = animal_features_averages[i]
        for j in range(len(features)):
            # if j == 12: # ignoring feature 12 (legs) leads to much better results
            #     continue
            different_from_average[i] += abs(averages[j] - features[j])

    return different_from_average.index(min(different_from_average)) + 1


def run_classifier():
    """ Run classifier for every animal (row) in training dataset """
    n_correct_classifications = 0
    correct = [0] * 7
    wrong = [0] * 7
    for animal, data_row in data_train.items():
        class_int = data_row[-1]
        animal_features = data_row[:-1]
        if classify_animal(*animal_features) == class_int:
            n_correct_classifications += 1
            correct[class_int-1] += 1
        else:
            print(animal, "WRONG:", classify_animal(*animal_features), "CORRECT:", class_int)
            wrong[class_int-1] += 1
    print(correct)
    print(wrong)
    print('Number of correct classifications: ' +
          f'{n_correct_classifications} / {len(data_train)}')


# RESULTS:
# data_train: 71/ 76
# data_test:  19/24

# Results while ignoring legs:
# data_train: 74/76
# data_test: 23/24

if __name__ == "__main__":
    # print_data()
    run_classifier()
