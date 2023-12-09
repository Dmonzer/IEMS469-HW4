

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#!nvidia-smi
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import  sklearn
from sklearn.model_selection import train_test_split
#import ray
import os
import math
import matplotlib.pyplot as plt
import tensorflow.keras.optimizers.schedules as schedules
import pickle
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
import random

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

originaldir = os.getcwd()
## Change to the directory of where you keep your files.
os.chdir('/nfs/home/dem1110/Assignment 4/')


train_X_all = np.load('train_X.npy', allow_pickle=True)
train_Y_all  = np.load('train_y.npy', allow_pickle=True)
test_X = np.load('test_X.npy', allow_pickle=True)
test_Y = np.load('test_y.npy', allow_pickle=True)


os.chdir(originaldir)
os.getcwd()

train_X_all.shape

train_Y_all.shape

test_X.shape

"""### Splitting the data into training and validation

"""

# Split the data into training and validation sets
x_train, x_val= train_test_split(
    train_X_all,
    test_size=0.2,
    random_state=42)

y_train, y_val= train_test_split(
    train_Y_all,
    test_size=0.2,
    random_state=42)


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
#combined_dataset = train_dataset.concatenate(val_dataset)

x_train.shape

len(train_dataset)

"""https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
"""

def macro_f1(y_true, y_pred):
    ## converting softmax probabilities to class predictions
    y_pred_classes = tf.argmax(y_pred, axis=1, output_type=tf.int32)
    y_true_classes = tf.cast(y_true, tf.int32)

    def _numpy_f1_score(y_true, y_pred):
        return f1_score(y_true, y_pred, average='macro')

    ## using tf.py_function to wrap the NumPy function
    ##so that it can be executed within TensorFlow's computational graph
    f1 = tf.py_function(_numpy_f1_score, (y_true_classes, y_pred_classes), tf.float64)
    return f1

###Testing one model.



"""### Running one model"""

#chromosome = [32, "relu"]
#activation = 'relu'
def get_model(lr, batch_size, activation):


    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation=activation),
        tf.keras.layers.Dense(10, activation="softmax")])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=[macro_f1])

    return model
##TESTING ONE MODEL
'''
run_model = get_model(initial_learning_rate, batch_size, activation)

run_model.fit(train_dataset.batch(batch_size), epochs=E, validation_data=val_dataset.batch(batch_size), verbose=1)
train_loss, train_f1 = run_model.evaluate(train_dataset.batch(batch_size), verbose=1)
val_loss, val_f1 = run_model.evaluate(val_dataset.batch(batch_size), verbose=1)
'''

"""### Hot decoding"""

batch_options_int = np.arange(16, 1025) # 16 to 1024
activation_options = ["relu","sigmoid", "tanh"]
batch_options_int_khara = list(batch_options_int)

encoder = OneHotEncoder(sparse_output=False)

batch_options_binary = encoder.fit_transform(np.array(batch_options_int).reshape(-1, 1)).astype(int)
activation_binary = encoder.fit_transform(np.array(activation_options).reshape(-1, 1)).astype(int)
activation_dict_bin = {}
batch_dict_bin ={}

activation_dict_str = {}
batch_dict_int ={}

#using dictionary to transform binary value to string or integer
for i in range(len(activation_options)):
    activation_dict_bin[tuple(activation_binary[i])] = activation_options[i]
    activation_dict_str[activation_options[i]] = tuple(activation_binary[i])

for j in range(len(batch_options_int)):
    batch_dict_bin[tuple(batch_options_binary[j])]=batch_options_int[j]
    batch_dict_int[batch_options_int[j]]=tuple(batch_options_binary[j])

activation_dict_str

activation_binary[1]

"""# Implementing GA

## Helper Functions

Given a dictionary with keys and values that are lists, this function finds the minimum
value in all of the lists (oldest chromosome who was inserted at the earliest generation)
"""

def key_with_min_value_and_remove(dictionary):
    min_value = 10000  # initializing min_value to positive infinity
    min_keys = []  # list to store keys with the minimum value

    # find the minimum element in a list in value item and corresponding key
    for key, value in dictionary.items():
        #print("value in age ditiondary value: ", value)
        if len(value) > 0:
            current_min = min(value)
            #print("current_min: ", current_min)
            if current_min < min_value:
                min_value = current_min
                min_keys = [key]
            elif current_min == min_value:
                min_keys.append(key)

    # randomly select one key from the keys with the minimum value
    if len(min_keys)>0:
        selected_key = random.choice(min_keys)
        dictionary[selected_key].remove(min_value)  # remove the minimum value from the list
        return selected_key
    else:
        return None  # return None if the dictionary is empty or all lists are empty

# Testing on dictionary
data = {'A': [3, 5, 9], 'B': [4, 2, 6],'C': [1, 8, 10],'D': [7, 1,1,1, 5]}

result_key = key_with_min_value_and_remove(data)
print("Key with the smallest value:", result_key)
print("Updated data dictionary after removal:", data)

#another example
result_key = key_with_min_value_and_remove(data)
print("Key with the smallest value:", result_key)
print("Updated data dictionary after removal:", data)




"""## Main Functions

This function finds the oldest chromosomes, removes them from the population list and give an updated list, and update the age_dictionary
"""

# take dictionary of age and remove old chromosomes
def remove_old_chromosomes_update_age(generation, population_list, children_list, age_dictionary, cut_off):
    #update both population_list and age dictionary
    nb_children = len(children_list)
    nb_population = len(population_list)
    if nb_children!=cut_off:
        print("MISMATCH ERROR between children size and cut-off")

    #age_dictionary willl include vectors as tuples, and they will be keys
    #the values include the list of generations where these vectors where added
    old_chromosomes = []
    for i in range(cut_off):
        # I want to find the key that has the smallest value in its list,
        #and if there are ties, pick any one key at random
        result_key = key_with_min_value_and_remove(age_dictionary)
        #print("result key: ", result_key)
        old_chromosomes.append(result_key)
        #print(len(old_chromosomes))
    #removing old chromosomes from population list
    for j in  range(len(old_chromosomes)):
        #print("an old chromosome: ", old_chromosomes[j])
        j_aslist = [np.array(old_chromosomes[j][0]), np.array(old_chromosomes[j][1])]
        '''
        if isinstance(old_chromosomes[j], tuple):
            print("old_chromosomes[j] is a tuple")
        else:
            print("old_chromosomes[j] is not a tuple")
        '''
        # Convert tuple elements to a NumPy array
        #result_array = np.array(my_tuple)
        #print(j_aslist)
        #if j_aslist in population_list:
        foundone = 0
        k = 0
        while foundone<1 and k<len(population_list):
            element = population_list[k]
            '''
            if isinstance(element, tuple):
                print("element is a tuple")
            else:
                print("element is not a tuple")
            '''
            if element[0]==old_chromosomes[j][0] and element[1] == old_chromosomes[j][1]:
                #old_chromosome.remove
                population_list.remove(element)
                #print("this is removed chromose: ", element)
                #print(len(population_list))
                foundone=1
            k+=1
    #adding age of children
    for k in range(nb_children):
        #child_tuple =  tuple([tuple(children_list[i][0]), tuple(children_list[i][1])]) #tuple(tuple(arr) if isinstance(arr, np.ndarray) else arr for arr in children_list[i])
        #print(type(child_tuple))
        #MAKE SURE children_list[i] IS TUPLE
        '''
        if isinstance(children_list[k], tuple):
            print("children_list[k] is a tuple")
        else:
            print("children_list[k] is not a tuple")
        '''
        if age_dictionary.get(children_list[k]) is None:
            age_dictionary[children_list[k]] = [generation]
        else:
            # Existing key found, append the generation
            age_dictionary[children_list[k]].append(generation)



    return population_list #, age_dictionary

#example
A = {tuple([1, 2, 3]):1, tuple([2, 3, 1]):2}
print(A[tuple([1,2,3])])

type(list(tuple([1,2,3])))

def roullete(population_list, fitness_dictionary, parents_subset):
    N = len(population_list)
    fitness_list = []
    for i in range(N):
        '''
        if isinstance(population_list[i], tuple):
            print("population_list[i] is a tuple")
        else:
            print("population_list[i] is not a tuple")
        '''
        fitness_list.append(fitness_dictionary[population_list[i]])
    deno = sum(fitness_list)
    if deno==0:
        deno =1
    # create a vector of probabilities for each element in population_list based on their fitness/deno
    probabilities = tf.convert_to_tensor([fitness / deno for fitness in fitness_list], dtype=tf.float32)

    '''
    dist = tfp.distributions.Categorical(probs=probabilities)
    selected_indices = dist.sample(sample_shape=parents_subset)
    selected_parents = [population_list[i] for i in selected_indices.numpy()]
    '''
     # Sample parents_subset elements from population_list based on their fitness
    selected_indices = random.choices(range(N), weights=probabilities, k=parents_subset)
    selected_parents = [population_list[j] for j in selected_indices]



    return selected_parents

def crossover(parents_list, parents_subset):
    m = len(parents_list)
    if m!=parents_subset:
        print("ERROR: mismatch in number of parents")
    nb_children = m/2.0
    children_list = []
    for i in range(int(nb_children)):
        ##randomly select 2 elements from parents_list using uniform distribution
        selected_parents = random.sample(parents_list, 2)

        ##perform a crossover between the two selected elements to create new children
        child1 = tuple([selected_parents[0][0], selected_parents[1][1]])
        child2 = tuple([selected_parents[1][0],  selected_parents[0][1]])

        ##append children to the children_list
        children_list.append(child1)
        children_list.append(child2)
        #print(child1)
        #print(child2)
    #print("crossover")
    return children_list

def save_checkpoints(metrics,history_best_dict,  iteration, population_list, fitness_dictionary, age_dictionary):


    #if isinstance(population_list, list):
        #population_list = np.array(population_list)

    #np.save(str(iteration)+'population_list.npy', population_list)

    #POPULATION
    with open('population_list.pickle', 'wb') as file:
        pickle.dump(population_list, file)

    #FITNESSS
    fitness_file = 'fitness_dictionary.pickle'
    with open(fitness_file, 'wb') as file:
        pickle.dump(fitness_dictionary, file)

    #AGE
    age_file = 'age_dictionary.pickle'
    with open(age_file, 'wb') as file:
        pickle.dump(age_dictionary, file)

    #MAX AND AVG
    fitness_list = []
    for i in range(len(population_list)):
        fitness_list.append(fitness_dictionary[population_list[i]])

    fitness_array = np.array(fitness_list)
    max_index = np.argmax(fitness_array)
    max_fitness = fitness_list[max_index]
    best_chromosome = population_list[max_index]
    avg_fitness = np.mean(fitness_array)

    best_batch = best_chromosome[0]
    best_act = best_chromosome[1]

    metrics.append([iteration,best_batch,best_act, max_fitness, avg_fitness ])
    np.save('metrics.npy', np.array(metrics))


    history_best_dict[generation] = [best_chromosome]

    history_file = 'history_best_dict.pickle'
    with open(history_file, 'wb') as file:
        pickle.dump(history_best_dict, file)



    return max_fitness, avg_fitness

'''
def swap_random_element(vector):
    #  the index of the element equal to 1
    vector = list(vector)
    index_of_one = vector.index(1)
    # another random index different from the index of 1
    random_index = random.choice([i for i in range(len(vector)) if i != index_of_one])
    # swap the elements at the two indices
    vector[index_of_one], vector[random_index] = vector[random_index], vector[index_of_one]

    return vector

'''
#swap mutation
def mutation(children_list, parents_subset, mutation_prob):
    m = len(children_list)
    '''
    if isinstance(children_list, list):
        print("in mutation, children_list is a list")
    else:
        print("in mutation, children_list is not a list")
    '''
    if m!=parents_subset:
        print("ERROR: mismatch in number of parents")


    for i in range(m):
        # for each child, toss a coin to decide to mutate the first element or not
        el1 = random.random()
        # toss a coing for the second element
        el2 = random.random()
        #print("children_list[i][0]: ", children_list[i][0], type(children_list[i][0]))
        #batch1 = batch_dict[tuple(children_list[i][0])]
        #activation1 = activation_dict[tuple(children_list[i][1])]
        #print("before mutation: batch = ", batch1, "activation = ", activation1)
        current_child = list(children_list[i])

        if el1 < mutation_prob:
        #do mutation to batch size
            index_of_one = batch_options_int_khara.index(children_list[i][0])
            batch_indices = [bi for bi in range(len(batch_options_int_khara)) if bi != index_of_one]
            # choose a random index from the list of available indices
            random_index_batch = random.choice(batch_indices)
            current_child[0] = batch_options_int_khara[random_index_batch]


        if el2 < mutation_prob:
        #do mutation to activation function
            index_of_act = activation_options.index(children_list[i][1])
            act_indices = [ai for ai in range(len(activation_options )) if ai != index_of_act ]
            # choose a random index from the list of available indices
            random_index_act = random.choice(act_indices)
            current_child[1] = activation_options[random_index_act]

        children_list[i] = tuple(current_child)
        '''
        if isinstance(children_list[i], tuple):
            print("in mutation, children_list[i] is a tuple")
        else:
            print("in mutation, children_list[i] is not a tuple")
        '''
    return children_list

def compute_fitness(lr, chromosome, fitness_dictionary):
    updated_fitness_dictionary = fitness_dictionary.copy()
    #if this chromosome has been tested before, skip this whole step
    '''
    if isinstance(chromosome, tuple):
        print("in compute_fitness, chromosome is a tuple")
    else:
        print("in compute_fitness, chromosome is not a tuple")
    '''
    if updated_fitness_dictionary.get(chromosome) is None or updated_fitness_dictionary.get(chromosome)==0:
    #chromosome hasn't been tested
    #run the model using the chromosome (make verbose =0)

        #translate bit elements to actual values of batch size and activation function
        batch_size = chromosome[0]
        activation = chromosome[1]

        #train model
        run_model = get_model(lr, batch_size, activation)

        run_model.fit(train_dataset.batch(batch_size), epochs=E, validation_data=val_dataset.batch(batch_size), verbose=0)
        #evaluate model
        train_loss, train_f1 = run_model.evaluate(train_dataset.batch(batch_size), verbose=0)
        val_loss, val_f1 = run_model.evaluate(val_dataset.batch(batch_size), verbose=0)
        #save fitness
        updated_fitness_dictionary [chromosome] = val_f1

        #print("batch size: ", batch_size, "activation: ", activation, "val_f1: ", val_f1)
    #else:
        #print("already computed")
        

    return updated_fitness_dictionary

#best_chromosome = track_best_chromosome(generation, history_best_dict, fitness_dictionary)[0]

##Create a function to check whether the fitness has converged.
###?????????????
##??????????????

"""# Initialization and Run Algorithm"""

## Hyperparameters
pop_size = 12
mutation_prob = 0.1
max_iterations = 200
subset_size = 4
lr = 0.01 #0.00001
E = 600


population = [tuple([random.choice(batch_options_int), random.choice(activation_options )]) for _ in range(pop_size)]

population

age_dict ={}


for sanfour in range (len(population)):
    chromosome = population[sanfour]
    #print(chromosome)
    if age_dict.get(chromosome) is None:
        age_dict[chromosome] = [0]
    else:
        age_dict[chromosome].append(0)

age_dict

fitness_dict = {vector: 0.0 for vector in population}



for asal in range(pop_size):
    chromosome = population[asal]
    fitness_dict = compute_fitness(lr, chromosome, fitness_dict)
    print(str(asal) + " is done")
fitness_dict

def plot_fitness_progress(generation, max_fitness, avg_fitness):
    generations_list = list(range(generation+1))

    plt.figure(figsize=(8, 6))
    plt.plot(generations_list, max_fitness, label='Max Fitness', color='green')
    plt.plot(generations_list, avg_fitness, label='Average Fitness', color='red')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Max and Average Fitness vs. Generation')
    plt.legend()
    plt.grid(True)
    plt.savefig("Max and Average Fitness vs. Generation.png")
    #plt.show()

"""## Reloading population and fitness and age due to interruption"""

'''
originaldir = os.getcwd()
os.chdir('/nfs/home/dem1110/Assignment 4/p1/run A/initialization- generation0')

with open('population_list.pickle', 'rb') as file:
    population = pickle.load(file)

# Load fitness dictionary from pickle
with open('fitness_dictionary.pickle', 'rb') as file:
    fitness_dict = pickle.load(file)

with open('age_dictionary.pickle', 'rb') as file:
    age_dict = pickle.load(file)
os.chdir(originaldir)
'''


"""# Running GA"""

#THe GA algorithm

generation = 0
max_fitness = []
avg_fitness = []
metrics = []
history_best_dict = {}
max_value, avg_value = save_checkpoints(metrics, history_best_dict, generation, population, fitness_dict, age_dict)
max_fitness.append(max_value)
avg_fitness.append(avg_value)
generation = 1
Not_Accomplished = avg_fitness[-1]<0.95 and max_fitness[-1]<1
while generation < max_iterations and Not_Accomplished: #fitness not converged # iteration <max:
    print("generation: ", generation)
    #get a subset of parents
    selected_parents = roullete(population, fitness_dict, subset_size)
    #print("number of selected parents: ", len(selected_parents))

    #run crossover
    children_list = crossover(selected_parents, subset_size)
    #print("number of children: ", len(children_list))

    #run mutatation
    children_list = mutation(children_list, subset_size, mutation_prob)

    #get fitness of new children
    for i in range (len(children_list)):

        fitness_dict = compute_fitness(lr, children_list[i], fitness_dict)

    #remove oldest parents but save copy of population

    population = remove_old_chromosomes_update_age(generation, population, children_list, age_dict, subset_size)

    #add children to rest of population
    for k in range (len(children_list)):
        population.append(children_list[k]) # = population + children_list

    #save metrics
    #print("updated pouplation size: ", len(population))


    #save checkpoints
    #if generation %1==0:
    max_value, avg_value = save_checkpoints(metrics, history_best_dict, generation, population, fitness_dict, age_dict)
    max_fitness.append(max_value)
    avg_fitness.append(avg_value)
    np.save('max_fitness_list.npy', np.array(max_fitness))
    np.save('avg_fitness_list.npy', np.array(avg_fitness))
    print("best chromosome of this generation: ", history_best_dict[generation])
    print("max fitness: ", max_fitness[-1], " avg fitness: ", avg_fitness[-1])
    #get average of last 5 highest fitness values and save it
    #compute the highest and lowest value of this list of last 5 values
    #if less than threshold, break loop
    if generation>4:
        plot_fitness_progress(generation, max_fitness, avg_fitness)

    generation+=1
    #not accomplished is now true, if one becomes false, it becomes false, and while loop breaks
    Not_Accomplished = avg_fitness[-1]<0.95 and max_fitness[-1]<1

    #population
    #age_dict

list(range(generation+1))

max_fitness

"""# After getting the Optimal Parameters

### Run the model on the full combined dataset
"""

last_gen = generation-1
plot_fitness_progress(last_gen, max_fitness, avg_fitness)

best_chromosome = history_best_dict[generation-1]

best_chromosome

batch_size_best = best_chromosome[0][0]
activation_best = best_chromosome[0][1]
final_model = get_model(lr, batch_size_best, activation_best)

combined_dataset = train_dataset.concatenate(val_dataset)
batched_combined_dataset  = combined_dataset .batch(batch_size_best )

history = final_model.fit(batched_combined_dataset, epochs=E, verbose=1)

loss = history.history['loss']

macro_f1_metric = history.history['macro_f1']  # Assuming accuracy is the metric you want to track

# Plotting epoch versus loss
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(loss) + 1), loss, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.legend()
plt.savefig("final training loss versus epoch.png")
plt.show()


# Plotting epoch versus accuracy
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(macro_f1_metric) + 1), macro_f1_metric, label='Training macro_f1_metric')
plt.xlabel('Epochs')
plt.ylabel('Macro-f1')
plt.title('Training macro_f1_metric per Epoch')
plt.legend()
plt.savefig("final training macro-f1 versus epoch.png")
plt.show()







#evaluate model
combined_train_loss, combined_train_f1 = final_model.evaluate(batched_combined_dataset, verbose=0)


print(f"Combined Train f1: {combined_train_f1 :.4f}")

"""### Evaluate on the Test Data"""

test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_Y))

batched_test_dataset = test_dataset.batch(batch_size_best)

test_loss, test_f1 = final_model.evaluate(batched_test_dataset)

print(f"Test F1: {test_f1:.4f}")
print(f"Test Loss: {test_loss:.4f}")









