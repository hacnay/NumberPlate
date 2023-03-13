import pickle
import SegmentCharacters

print("Loading model")
filename = './finalized_model.sav'
model = pickle.load(open(filename, 'rb'))
print('Model loaded. Predicting characters of number plate')

# Predict characters of number plate
classification_result = model.predict(SegmentCharacters.characters)
print('Classification result')
print(classification_result)

# Convert classification result to plate string
plate_string = ''.join(classification_result)
print('Predicted license plate')
print(plate_string)

# Sort characters in the right order
column_list_copy = SegmentCharacters.column_list[:]
SegmentCharacters.column_list.sort()
rightplate_string = ''.join([plate_string[column_list_copy.index(each)] for each in SegmentCharacters.column_list])

print('License plate')
print(rightplate_string)
