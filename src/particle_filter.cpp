/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// Set global params
#define NUM_PARTICLES   50
#define EPS             0.001

void dbg_pp (Particle &particle) {
    // Print your samples to the terminal.
	cout << "Sample: " << particle.id << " " << particle.x  <<" " << particle.y << " " << particle.theta << endl;
}

void dbg_po (LandmarkObs &obs) {
    // Print your samples to the terminal.
	cout << "Observation: " << obs.id << " " << obs.x  <<" " << obs.y << endl;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	// Set number of Particles
	num_particles = NUM_PARTICLES;
	
	// Resize the vectors of weights/particles
	weights.resize(num_particles);
	particles.resize(num_particles);
	
	// Create a normal (Gaussian) distribution for x,y & theta.
	random_device rd;
    default_random_engine gen(rd());
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	
	// Initialize the Particle positions (x,y) and yaw(theta) as samples from
	// Initial GPS Gaussian distribution and weights to 1.0
	for (int i = 0; i < particles.size(); ++i) {
		particles[i].id = i;
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		particles[i].weight = 1.0;
		// Print your samples to the terminal.
		//dbg_pp(particles[i]);
	}
    
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	// Update each Particle position based on given params and then add random Gaussian noise
	// with mean based on updated particle position and std_dev of the measurements
    random_device rd;
	default_random_engine gen(rd());
	
	// Update Particle positions
	double velocity_dt = velocity * delta_t;
	double yaw_rate_dt = yaw_rate * delta_t;
	
	for (int i = 0; i < particles.size(); ++i) {
		// If yaw_rate is very small then consider particle moving straight
		if (fabs(yaw_rate) < EPS) {
			particles[i].x += velocity_dt * cos(particles[i].theta);
			particles[i].y += velocity_dt * sin(particles[i].theta);
		} 
		else { // Update/predict position with yaw_rate consideration
		    double velocity_yaw_rate = velocity/yaw_rate;
		    double theta_dt = particles[i].theta + yaw_rate_dt;
			particles[i].x += velocity_yaw_rate * (sin(theta_dt) - sin(particles[i].theta));
			particles[i].y += velocity_yaw_rate * (cos(particles[i].theta) - cos(theta_dt));
			particles[i].theta = theta_dt;
		}	
		
		// Add random Gaussian noise to the predicted position with sensor measurement variances (std_pos)
		normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
		normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
		normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);
		
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		
		// Print your samples to the terminal.
		//dbg_pp(particles[i]);
	} 
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	// Check distance for each observed measurement from all predicted landmarks
	//double xx, yy, oxx, oyy;
	for (int i = 0; i < observations.size(); i++) {
		double dist_min = numeric_limits<double>::max();
		int associated_landmark_id = -1;
		double obx = observations[i].x;
		double oby = observations[i].y;
	
		for (int j = 0; j < predicted.size(); j++) {
			double dist_landmark = dist(obx, oby, predicted[j].x, predicted[j].y);
			if (dist_landmark < dist_min) {
				associated_landmark_id = predicted[j].id;
				dist_min = dist_landmark;
				//xx = predicted[j].x;
				//yy = predicted[j].y;
				//oxx = observations[i].x;
				//oyy = observations[i].y;
				// Store the difference in x & y to be used for Weigt calculations effectively
				observations[i].x  = fabs(obx - predicted[j].x);
				observations[i].y  = fabs(oby - predicted[j].y);
			}
		}
		if (associated_landmark_id == -1) {
			cout << "Error: No associated landmark found" << endl;
		}
		observations[i].id = associated_landmark_id;

        //cout << "Observation:  " << observations[i].id << " " << oxx << " " << oyy << endl;		
        //cout << "Predicted:    " << associated_landmark_id << " " << xx << " " << yy << endl;	
		
	}		
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	double weight_sum = 0.0;
	
	//cout << "size of map landmarks: " << map_landmarks.landmark_list.size() << endl;
	
	for (int i = 0; i < particles.size(); i++) {
		// Particle params
		double par_x = particles[i].x;
		double par_y = particles[i].y;
		double par_t = particles[i].theta;

		// Find Landmark positions within sensor_range of the Particle
		vector<LandmarkObs> within_range_landmarks;
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			double mx = map_landmarks.landmark_list[j].x_f;
			double my = map_landmarks.landmark_list[j].y_f;
			if ( dist(par_x, par_y, mx, my) <= sensor_range ) {
				within_range_landmarks.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i, mx, my});
			}
		}
		//cout << "In Range landmarks size: " << within_range_landmarks.size() << endl;
		// Transform Observed measurements into Map co-ordinates
		vector<LandmarkObs> mapped_observations;
		for (int j = 0; j < observations.size(); j++) {
			double mapx = cos(par_t) * observations[j].x - sin(par_t) * observations[j].y + par_x;
			double mapy = sin(par_t) * observations[j].x + cos(par_t) * observations[j].y + par_y;
			mapped_observations.push_back(LandmarkObs{observations[j].id, mapx, mapy});
		}
        //cout << "mapped observations size: " << mapped_observations.size() << endl;
		// Associate Mapped Observations to Landmarks within Range
		dataAssociation(within_range_landmarks, mapped_observations);
		
		// Calculate Multivariate Gaussian probability for each mapped observations
		particles[i].weight = 1.0;
		for (int j = 0; j < mapped_observations.size(); j++) {
			double mvg_prob;
			double diffX = mapped_observations[j].x; // Difference with associated Landmark is stored in x
			double diffY = mapped_observations[j].y; // Difference with associated Landmark is stored in y
			
			if (mapped_observations[j].id != -1) {
			   	double mvg_prob = (1/(2*M_PI*std_landmark[0]*std_landmark[1])) * exp(-((diffX*diffX)/(2*std_landmark[0]*std_landmark[0]) + (diffY*diffY)/(2*std_landmark[1]*std_landmark[1])));
                if (mvg_prob < EPS) { mvg_prob = EPS; }
				//cout <<"mvg_prob: " << mvg_prob << endl;
				particles[i].weight *= mvg_prob;
			}
            			
        }
		weight_sum += particles[i].weight;
		//cout << "Particles weight: " << particles[i].weight << endl;
	}
	
	// Normalize the weights
	
	for (int i = 0; i < particles.size(); i++) {
		particles[i].weight /= weight_sum;
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	random_device rd;
	default_random_engine gen(rd());
	
	vector<Particle> resampled_particles;
	resampled_particles.resize(particles.size());
	
	discrete_distribution<int> dist_resample(weights.begin(), weights.end());
	for (int i = 0; i < particles.size(); i++) {
		resampled_particles[i] = particles[dist_resample(gen)];
	}
	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
