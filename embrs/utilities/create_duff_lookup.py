import csv
import pickle
import json

def create_lookup_dict(csv_filename):
    lookup_dict = {}
    with open(csv_filename, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Using the first column ("FCCS") as key and "Duff" column as value
            key = int(row['FCCSID'].strip())
            value = float(row['Duff'].strip())
            lookup_dict[key] = value
    return lookup_dict

def save_dict_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_dict_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_dict_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    csv_filename = '/Users/rjdp3/Research/Code/embrs/FCCS_test/fccs_table.csv'         # CSV file containing your data
    pickle_filename = 'duff_loading.pkl' # File to store the dictionary as a pickle
        
    # Create the lookup dictionary from the CSV file
    lookup_dict = create_lookup_dict(csv_filename)
    
    # Save the dictionary using pickle
    save_dict_pickle(lookup_dict, pickle_filename)
    print(f"Lookup dictionary saved to {pickle_filename}")
    
