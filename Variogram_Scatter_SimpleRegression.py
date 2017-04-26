#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 19:09:43 2017

@author: neelabhpant
"""

""" This is a temporary code.
    This code will soon be replaced with the implementation of DISTANCE calculation
    and VARIOGRAMS, SCATTERPLOT AND SIMPLE REGRESSION will be under 
    respective dedicated functions """


import Statistics
from Statistics import covariance, variance, mean, de_mean
import matplotlib.pyplot as plt
from scipy.spatial import distance

labels = ['CA-TX', 'CA-IL', 'CA-AR', 'CA-CO', 'CA-WA', 'TX-IL', 'TX-AR', 'TX-CO', 'TX-WA', 'IL-AR', 'IL-CO', 'IL-WA', 'AR-CO', 'AR-WA', 'CO-WA']
Euclidean = [24.93, 33.99, 28.99, 18.44, 11.19, 12.84, 5.44, 8.01, 26.066, 7.51, 15.72, 30.97, 10.72, 28.299, 18.09]
Attribute = [1.414, 1.732, 2, 1, 2.23, 1, 1.414, 1, 1.732, 1, 1.414, 1.414, 1.732, 1, 2]
Elevation = [34.64, 47.95, 47.43, 62.44, 34.64, 33.166, 32.40, 71.41, 0, 7.07, 78.74, 33.166, 78.42, 32.40, 71.41]

plt.subplots_adjust(bottom = 0.1)
plt.scatter(Euclidean, Attribute, marker='o', cmap=plt.get_cmap('Spectral'))
for label, x, y in zip(labels, Euclidean, Attribute):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.xlabel('Euclidean Distance')
plt.ylabel('Time Difference')
plt.show()



plt.subplots_adjust(bottom = 0.1)
plt.scatter(Euclidean, Elevation, marker='o', cmap=plt.get_cmap('Spectral'))
for label, x, y in zip(labels, Euclidean, Elevation):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.xlabel('Pairwise Normalized Euclidean Distance')
plt.ylabel('Square Root of Absolute Difference of Elevation in Feet')
plt.show()



state = ['CA', 'TX', 'IL', 'AR', 'CO', 'WA']
labels = ['CA-TX', 'CA-IL', 'CA-AR', 'CA-CO', 'CA-WA', 'TX-IL', 'TX-AR', 'TX-CO', 'TX-WA', 'IL-AR', 'IL-CO', 'IL-WA', 'AR-CO', 'AR-WA', 'CO-WA']
lat = [37.693339, 32.766721, 41.604031, 35.855370, 38.081431, 48.319464]
lon = [-121.659263, -97.216667, -87.892864, -92.728843, -103.218875, -118.135100]
lat_lon = [[37.693339,-121.659263], [32.766721, -97.216667], [41.604031, -87.892864], [35.855370, -92.728843], [38.081431, -103.218875], [48.319464, -118.135100]]
Period = [2,4,5,6,3,7]
Mean_Elevation = [2900, 1700, 600, 650, 6800, 1700]
Mean_neighbor_attribute_value = [2290, 2530, 2750, 2740, 1510, 2530]
dist = []
for i in range(5):
    for j in range(i+1, 6):
        d = distance.euclidean(lat_lon[i], lat_lon[j])
        dist.append(d)
    
    
plt.subplots_adjust(bottom = 0.1)
plt.scatter(Mean_Elevation, Mean_neighbor_attribute_value, marker='o', cmap=plt.get_cmap('Spectral'))
for label, x, y in zip(state, Mean_Elevation, Mean_neighbor_attribute_value):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.1),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.xlabel('Attribute Values')
plt.ylabel('Average Attribute Values Over Neighborhood')
plt.show()


def predict(alpha, beta, x_i):
    return (beta*x_i)+alpha

def error(alpha, beta, x_i, y_i):
    return y_i - predict(alpha, beta, x_i)

def sum_of_sq_error(alpha, beta, x, y):
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))

def least_squares_fit(x, y):
    beta = covariance(x, y) / variance(x)
    alpha = alpha = mean(y) - (beta * mean(x))
    return alpha, beta

def total_sum_of_squares(y):
    return sum(y_i**2 for y_i in de_mean(y))

def r_squared(alpha, beta, x, y):
    return 1.0 - (sum_of_sq_error(alpha, beta, x, y) / total_sum_of_squares(y))

alpha, beta = least_squares_fit(Mean_Elevation, Mean_neighbor_attribute_value)
Beta_alpha = [((beta*i) + alpha) for i in Mean_Elevation]

plt.subplots_adjust(bottom = 0.1)
plt.scatter(Mean_Elevation, Mean_neighbor_attribute_value, marker='o', cmap=plt.get_cmap('Spectral'))
for label, x, y in zip(state, Mean_Elevation, Mean_neighbor_attribute_value):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.2),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.xlabel('Attribute Values')
plt.ylabel('Average Attribute Values Over Neighborhood')
plt.plot(Mean_Elevation, Beta_alpha, '-')
plt.show()