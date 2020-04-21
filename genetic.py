from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
import skimage.draw as draw
from numba import jit, cuda
import time

shape_size = shape_y = 512
max_color = 255
goal_image = np.array(Image.open("Ready pictures/pathfinder_r.jpg"))
population = []
number_of_members_in_population = 5
best_cand = []
fitness_scores = []


def save_image(array, number):
    image = []
    for i in range(shape_size):
        row = []
        for y in range(shape_size):
            pixel = [0, 0, 0]
            row.append(pixel)
        image.append(row)
    image = np.array(image)

    for gene in array:
        row, column, r, g, b = gene[0], gene[1], gene[2], gene[3], gene[4]
        image[row, column] = r, g, b
    plt.imshow(image)
    plt.axis('off')
    plt.savefig("F:/results/"+str(number)+".jpg", dpi=300, bbox_inches='tight', pad_inches=0)


def show_from_array(array_of_genes):
    image = []
    for i in range(shape_size):
        row = []
        for y in range(shape_size):
            pixel = [0, 0, 0]
            row.append(pixel)
        image.append(row)
    image = np.array(image)

    for gene in array_of_genes:
        row, column, r, g, b = gene[0], gene[1], gene[2], gene[3], gene[4]
        image[row, column] = r, g, b
    show(image)


def show(picture):
    plt.imshow(picture)
    plt.axis('off')
    plt.show()


def rand_coord():
    return np.random.randint(1, shape_size)


def rand(min, max):
    return np.random.randint(min, max)


def rand_color():
    r, g, b = rand(1, max_color+1), rand(1, max_color + 1), rand(1, max_color + 1)
    return r, g, b


def create_new_gene():
    end_row, end_column = rand_coord(), rand_coord()
    start_row, start_column = rand_coord(), rand_coord()
    row, column = draw.line(start_row, start_column, end_row, end_column)
    r, g, b = rand_color()
    return np.array([row, column, r, g, b])

# @jit(target ="cuda")
def fitness_score(candidate):
    score = 0
    for gene in candidate:
        overall_error = 0
        # gene_color = [gene[2], gene[3], gene[4]]
        gene_row = gene[0]
        gene_column = gene[1]
        length_gene = len(gene_row)
        color_sum = [0, 0, 0]
        # need to be calculated
        r, g, b = gene[2], gene[3], gene[4]
        for i in range(length_gene):
            color_sum[0] += int(goal_image[gene_row[i], gene_column[i], 0])
            color_sum[1] += int(goal_image[gene_row[i], gene_column[i], 1])
            color_sum[2] += int(goal_image[gene_row[i], gene_column[i], 2])
        color_sum[0] = int(color_sum[0]/length_gene)
        color_sum[1] = int(color_sum[1]/length_gene)
        color_sum[2] = int(color_sum[2]/length_gene)
        overall_error += abs(r - color_sum[0]) + abs(g - color_sum[1]) + abs(b - color_sum[2])
        score += overall_error
    return score


def select_parents(array_of_scores):
    arr_of_scores = np.copy(array_of_scores)
    first_parent = np.argmin(arr_of_scores)
    arr_of_scores[first_parent] = 300000000
    second_parent = np.argmin(arr_of_scores)
    return [first_parent, second_parent]


def crossover(parents, population):
    length = len(population[parents[0]])
    cross_point = np.random.randint(1, length-1)
    first_parent = population[parents[0]]
    second_parent = population[parents[1]]
    offspring_1 = np.concatenate((first_parent[0:cross_point], second_parent[cross_point:length]))
    offspring_2 = np.concatenate((second_parent[0:cross_point], first_parent[cross_point:length]))
    return np.concatenate((population, np.array([offspring_1, offspring_2])))


def mutation(population):
    l = len(population[0])
    for chromo in population:
        for i in range(rand(0, len(population[0]))):
            l = len(population[0])
            random_gene = rand(0, l)
            for y in range(3):
                rand_num = rand(-30, 31)
                gene_color = (chromo[random_gene, y+2] + rand_num)
                chromo[random_gene, y+2] = 255 if gene_color > 255 else 0 if gene_color < 0 else gene_color


def create_chromo(startx, finishx, starty, finishy, step):
    radius = 4
    # width of cell = radius*2 - 1
    chromo = []
    # 16 то есть 4 на 4 работало хорошо
    # 2 by 2 сделать надо
    for i in range(startx, finishx, step):
        for y in range(starty, finishy, step):
            rr, cc = draw.circle(i, y, radius)
            r, g, b = rand_color()
            chromo.append(np.array([rr, cc, r, g, b]))
    return np.array(chromo)


def delete_worst_cand():
    global fitness_scores
    global population
    max_score_index = np.argmax(fitness_scores)
    population1 = []
    for i in range(len(population)):
        if i != max_score_index:
            population1.append(population[i])
    population = population1
    fitness_scores = np.delete(fitness_scores, max_score_index)


accept_bound = 70  # works well for 2 by 2
image_number = 1
overall_time = 0
for i in range(4, 512, 16):
    for y in range(4, 512, 16):
        population = []
        for n in range(number_of_members_in_population):
            chromo = create_chromo(i, i+16, y, y+16, 8)
            population.append(chromo)

        generation_number = 1
        best_chromo = []
        best_fit_score = 3000000
        start_time = time.time()
        while best_fit_score > accept_bound:
            fitness_scores = np.array([])
            for member in population:
                fit_score = fitness_score(member)
                fitness_scores = np.append(fitness_scores, fit_score)
            fitness_scores = np.array(fitness_scores)
            delete_worst_cand()
            delete_worst_cand()
            best_fit_score_index = np.argmin(fitness_scores)
            best_fit_score = fitness_scores[best_fit_score_index]
            # if generation_number % 100000 == 0:
            #     # save_image(population[best_fit_score_index], generation_number)
            #     print()
            #     print("generation " + str(generation_number))
            #     print(best_fit_score)
            #     print(fitness_scores)

            # select two parents (indexes in population)
            parents = select_parents(fitness_scores)
            # make crossover between them
            population = crossover(parents, population)
            # make mutation
            mutation([population[-1], population[-2]])
            generation_number += 1
            best_chromo = population[best_fit_score_index]
        end_time = time.time()
        time_taken = end_time - start_time
        overall_time += time_taken
        if len(best_cand) == 0:
            best_cand = best_chromo
        else:
            best_cand = np.concatenate((best_cand, best_chromo))
        if image_number % 32 == 0:
            save_image(best_cand, image_number)
            print("time taken for %d is %f" % (image_number, time_taken))
        image_number += 1

print("overall time %f minutes" % (overall_time/60))
# 1) Randomly initialize populations p
# 2) Determine fitness of population
# 3) Untill convergence repeat:
#       a) Select parents from population
#       b) Crossover and generate new population
#       c) Perform mutation on new population
#       d) Calculate fitness for 