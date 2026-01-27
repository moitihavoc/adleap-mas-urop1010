from ast import literal_eval
import numpy as np
import re

def read(nexp, method, path, env, 
    columns = ['it','reward','time','nrollouts','nsimulations','coop'],
    time_constrained=False, max_time_minutes=10):

    results = []
    for exp in range(nexp):
        # preparing results dict
        results.append({})
        for column in columns:
            results[-1][column] = []

        # reading the data
        with open(path+method+'_'+env+'_'+str(exp)+'.csv','r') as resultfile:
            count, running_time = 0, 0.0
            for line in resultfile:
                if count > 0:
                    fcolumns = line.split(';')
                    for i in range(len(columns)):
                        results[exp][columns[i]].append(float(fcolumns[i]))
                    
                    if time_constrained:
                        running_time += float(fcolumns[2])
                        if running_time > max_time_minutes*60:
                            break
                count += 1
    return results

def read_estimation(nexp, method, estimation, path, env, 
    columns = ['it','reward','time','nrollouts','nsimulations',\
     'typeestimation','typeestimation_err','parameterestimation','parameterestimation_err','memory'],
    time_constrained=False, max_time_minutes=10):

    results = []
    for exp in range(nexp):
        # preparing results dict
        results.append({})
        for column in columns:
            results[-1][column] = []

        # reading the data
        with open(path+method+'_'+estimation+'_'+env+'_'+str(exp)+'.csv','r') as resultfile:
            count, running_time = 0, 0.0
            for line in resultfile:
                if count > 0:
                    fcolumns = line.split(';')
                    for i in range(len(columns)):
                        new_string = fcolumns[i]
                        if new_string != 'NotImplemented':
                            if columns.index('parameterestimation_err') == i:
                                new_string = re.sub(r"\[(.*?)\]", r"\1", new_string)

                            if 'array' in new_string or 'float' in new_string:
                                new_string = extract_numbers_from_string(new_string)
                            else:
                                new_string = literal_eval(new_string)
                            results[exp][columns[i]].append(new_string)
                        else:
                            results[exp][columns[i]].append(None)
                    
                    if time_constrained:
                        running_time += float(fcolumns[2])
                        if running_time > max_time_minutes*60:
                            break
                count += 1
    return results

def extract_numbers_from_string(text):
    """
    Recursively extracts all numbers from a string representation of nested lists and np.array(...) formats.
    Preserves the original nested list structure.
    """
    # Replace np.array(...) with a Python list format
    if 'array' in text:
        text = re.sub(r"array\((.*?)\)", r"\1", text)
    if 'np.float' in text:
        text = re.sub(r"np\.float64\((.*?)\)", r"\1", text)
    
    # Convert the string to a nested Python list
    nested_structure = eval(text)

    # Extract numbers and preserve the structure
    def extract_numbers(structure):
        if isinstance(structure, list):
            return [extract_numbers(item) if isinstance(item, list) else float(item) for item in structure]
        return structure
    
    return extract_numbers(nested_structure)