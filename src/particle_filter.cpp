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
static default_random_engine generator;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 10;

	normal_distribution<double> Noise_x(0, std[0]);
	normal_distribution<double> Noise_y(0, std[1]);
 	normal_distribution<double> Noise_theta(0, std[2]);

	for (int i=0; i<num_particles; i++ ){
		Particle p;
		p.id = i;
		p.x = x + Noise_x(generator);
		p.y = y + Noise_y(generator);
		p.theta = theta + Noise_theta(generator);
		p.weight = 1.0;
		particles.push_back(p);
	}
	is_initialized=true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	normal_distribution<double> Noise_x(0, std_pos[0]);
	normal_distribution<double> Noise_y(0, std_pos[1]);
 	normal_distribution<double> Noise_theta(0, std_pos[2]);
	for (int i=0; i<num_particles; i++){
		if ( fabs(yaw_rate) > 0.00001 ) {
			particles[i].x += velocity/yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			particles[i].y += velocity/yaw_rate * (-cos(particles[i].theta + yaw_rate*delta_t) + cos(particles[i].theta));
			particles[i].theta += yaw_rate * delta_t;
		} else {
			particles[i].x += velocity* cos(particles[i].theta) * delta_t;
			particles[i].y += velocity* sin(particles[i].theta) * delta_t;
		}
		particles[i].x += Noise_x(generator);
		particles[i].y += Noise_y(generator);
		particles[i].theta += Noise_theta(generator);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i=0; i<observations.size(); i++) {
		LandmarkObs observation = observations[i];
	  double min_distance = numeric_limits<double>::max();
	  int id_with_min_distance = -1;

	  for (int j=0; j<predicted.size(); j++) {
	  	LandmarkObs prediction = predicted[j];
	   	double dx = observation.x - prediction.x;
			double dy = observation.y - prediction.y;
			double dist = pow(dx,2) + pow(dy,2);
    		dist = sqrt(dist);
    		if (dist < min_distance) {
				min_distance = dist;
				id_with_min_distance = prediction.id;
			}
		}
		observations[i].id = id_with_min_distance;
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
		
  for (int i=0; i<num_particles; i++){
		
		double p_x = particles[i].x;
    double p_y = particles[i].y;
    double particle_theta = particles[i].theta;
		
	  vector<LandmarkObs> sel_predictions;
		
		for (int j=0; j< map_landmarks.landmark_list.size(); j++) {
			float lx = map_landmarks.landmark_list[j].x_f;
			float ly = map_landmarks.landmark_list[j].y_f;	
			int lid = map_landmarks.landmark_list[j].id_i;	
			double dx = lx - particles[i].x;
			double dy = ly - particles[i].y;
			double dist = pow(dx,2) + pow(dy,2);
    	if (sqrt(dist) <= sensor_range){
			//if (fabs(lx - p_x) <= sensor_range && fabs(ly - p_y) <= sensor_range) {
				LandmarkObs obs = LandmarkObs{lid, lx, ly};
				sel_predictions.push_back(obs);
			}
		}
		
		//convert car co-ordinates to map co-ordinates
		vector<LandmarkObs> t_observations;
		for (int j=0; j<observations.size(); j++) {
			double transformed_x = cos(particle_theta) * observations[j].x - sin(particle_theta) * observations[j].y + particles[i].x;
		  double transformed_y = sin(particle_theta) * observations[j].x + cos(particle_theta) * observations[j].y + particles[i].y;
		  t_observations.push_back(LandmarkObs{observations[j].id, transformed_x, transformed_y });
		}

		dataAssociation(sel_predictions, t_observations);
		
		particles[i].weight = 1.0;
		 
		for (int j=0; j<t_observations.size(); j++) {
			int associated_prediction = t_observations[j].id;
			double pr_x, pr_y;
			for (int k=0; k<sel_predictions.size(); k++) {
		    if (sel_predictions[k].id == associated_prediction) {
		      pr_x = sel_predictions[k].x;
		      pr_y = sel_predictions[k].y;
		    }
		  }
      double s_x = std_landmark[0];
		  double s_y = std_landmark[1];
		  double o_x = t_observations[j].x;
      double o_y = t_observations[j].y;
		  particles[i].weight *= ( 1/(2*M_PI*s_x*s_y)) * exp( -( pow(pr_x-o_x,2)/(2*pow(s_x, 2)) + (pow(pr_y-o_y,2)/(2*pow(s_y, 2))) ) );
 		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> resampled_particles;
	double beta = 0.0;
	uniform_int_distribution<int> udist(0, num_particles-1);
	int index = udist(generator);

	vector<double> weights;
	for (int i = 0; i < num_particles; i++) {
	  weights.push_back(particles[i].weight);
  }

  double max_weight = *max_element(weights.begin(), weights.end());
	uniform_real_distribution<double> urealdist(0.0, max_weight);

	for (int i = 0; i < num_particles; i++) {
		beta += urealdist(generator) * 2.0  * max_weight;
	  while (beta > weights[index]) {
	    beta -= weights[index];
	    index = (index + 1) % num_particles;
	  }
	  resampled_particles.push_back(particles[index]);
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