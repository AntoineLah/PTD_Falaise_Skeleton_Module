//#include "/sps/nemo/scratch/lahaie/mydict.cxx"
#include <bayeux/dpp/chain_module.h>
#include <datatools/clhep_units.h>
#include <datatools/utils.h>
#include <bayeux/geomtools/blur_spot.h>


#include <falaise/snemo/datamodels/particle_track_data.h>
#include <falaise/snemo/datamodels/tracker_trajectory_solution.h>
#include <falaise/snemo/datamodels/tracker_trajectory_data.h>
#include <falaise/snemo/datamodels/tracker_clustering_data.h>
#include <falaise/snemo/datamodels/base_trajectory_pattern.h>
#include <falaise/snemo/datamodels/tracker_trajectory.h>
#include <falaise/snemo/datamodels/particle_track.h>
#include <falaise/snemo/datamodels/tracker_cluster.h>
#include <falaise/snemo/datamodels/calibrated_calorimeter_hit.h>
#include <falaise/snemo/datamodels/vertex_utils.h>
#include <falaise/snemo/datamodels/calibrated_tracker_hit.h>
#include <falaise/snemo/datamodels/precalibrated_data.h>
#include <falaise/snemo/datamodels/precalibrated_tracker_hit.h>
#include <falaise/snemo/datamodels/precalibrated_calorimeter_hit.h>
#include <falaise/snemo/datamodels/udd_utils.h>
#include <falaise/snemo/datamodels/unified_digitized_data.h>
#include <falaise/snemo/datamodels/calorimeter_digitized_hit.h>
#include <falaise/snemo/datamodels/tracker_digitized_hit.h>
#include <falaise/snemo/datamodels/geomid_utils.h>
#include <falaise/snemo/datamodels/timestamp.h>
#include <falaise/snemo/datamodels/track_fitting_utils.h>

#include "TFile.h"
#include "TTree.h"
#include <vector>
#include <string>
#include "TLatex.h"
#include "TVector3.h"
#include "TSystem.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>
#include <cmath>
#include <climits>
#include <chrono>
#include <iomanip>
#include <limits>
#include <numeric>
using namespace std;




// Helpe function to calculate spatial distance :) 
double calculateDistance(double x1, double y1, double z1, double x2, double y2, double z2) {
    return std::sqrt(std::pow(x1 - x2, 2) +
                     std::pow(y1 - y2, 2) +
                     std::pow(z1 - z2, 2));
}


struct TrackMatch {
    int elec_idx;         // Index of the electron candidate in your arrays
    int und_idx;          // Index of the UND candidate in your arrays
    int elec_fit_idx;     // Index of the fit solution for the electron
    int und_fit_idx;      // Index of the fit solution for the UND
    double spatial_distance; // Calculated spatial distance between the two tracks (vertex or end points)
    double time_difference;  // Time difference between the two tracks
    std::string closest_point; // Closest point (start or end)
};

// Helper function to merge two vectors
template <typename T>
std::vector<T> merge_vectors(const std::vector<T>& a, const std::vector<T>& b) {
    std::vector<T> result = a;
    result.insert(result.end(), b.begin(), b.end());
    return result;
}

// help function to find the minimum distance between two tracks ! 
double find_min_distance_between_tracks(int elec_idx, int und_idx, 
	const vector<vector<double>>* x_start_per_elec_cluster,
	const vector<vector<double>>* y_start_per_elec_cluster,
	const vector<vector<double>>* z_start_per_elec_cluster,
	const vector<vector<double>>* x_end_per_elec_cluster,
	const vector<vector<double>>* y_end_per_elec_cluster,
	const vector<vector<double>>* z_end_per_elec_cluster,
	const vector<vector<double>>* x_start_per_UND_cluster,
	const vector<vector<double>>* y_start_per_UND_cluster,
	const vector<vector<double>>* z_start_per_UND_cluster,
	const vector<vector<double>>* x_end_per_UND_cluster,
	const vector<vector<double>>* y_end_per_UND_cluster,
	const vector<vector<double>>* z_end_per_UND_cluster,
	int elec_fit_idx,
	int und_fit_idx) 
{
		// Calculate distances between all possible combinations of endpoints
		double d1 = calculateDistance(
		x_start_per_elec_cluster->at(elec_idx).at(elec_fit_idx),
		y_start_per_elec_cluster->at(elec_idx).at(elec_fit_idx),
		z_start_per_elec_cluster->at(elec_idx).at(elec_fit_idx),
		x_start_per_UND_cluster->at(und_idx).at(und_fit_idx),
		y_start_per_UND_cluster->at(und_idx).at(und_fit_idx),
		z_start_per_UND_cluster->at(und_idx).at(und_fit_idx)
		);

		double d2 = calculateDistance(
		x_start_per_elec_cluster->at(elec_idx).at(elec_fit_idx),
		y_start_per_elec_cluster->at(elec_idx).at(elec_fit_idx),
		z_start_per_elec_cluster->at(elec_idx).at(elec_fit_idx),
		x_end_per_UND_cluster->at(und_idx).at(und_fit_idx),
		y_end_per_UND_cluster->at(und_idx).at(und_fit_idx),
		z_end_per_UND_cluster->at(und_idx).at(und_fit_idx)
		);

		double d3 = calculateDistance(
		x_end_per_elec_cluster->at(elec_idx).at(elec_fit_idx),
		y_end_per_elec_cluster->at(elec_idx).at(elec_fit_idx),
		z_end_per_elec_cluster->at(elec_idx).at(elec_fit_idx),
		x_start_per_UND_cluster->at(und_idx).at(und_fit_idx),
		y_start_per_UND_cluster->at(und_idx).at(und_fit_idx),
		z_start_per_UND_cluster->at(und_idx).at(und_fit_idx)
		);

		double d4 = calculateDistance(
		x_end_per_elec_cluster->at(elec_idx).at(elec_fit_idx),
		y_end_per_elec_cluster->at(elec_idx).at(elec_fit_idx),
		z_end_per_elec_cluster->at(elec_idx).at(elec_fit_idx),
		x_end_per_UND_cluster->at(und_idx).at(und_fit_idx),
		y_end_per_UND_cluster->at(und_idx).at(und_fit_idx),
		z_end_per_UND_cluster->at(und_idx).at(und_fit_idx)
		);

		// Return the minimum distance
		return std::min({d1, d2, d3, d4});
}

// help function to find the minimum distance between electron crossing and UND tracks
double find_min_distance_between_crossing_and_UND(
    int crossing_idx, int und_idx,
    const vector<vector<double>>* x_start_per_elec_crossing_4_alpha,
    const vector<vector<double>>* y_start_per_elec_crossing_4_alpha,
    const vector<vector<double>>* z_start_per_elec_crossing_4_alpha,
    const vector<vector<double>>* x_end_per_elec_crossing_4_alpha,
    const vector<vector<double>>* y_end_per_elec_crossing_4_alpha,
    const vector<vector<double>>* z_end_per_elec_crossing_4_alpha,
    const vector<vector<double>>* x_start_per_UND_cluster,
    const vector<vector<double>>* y_start_per_UND_cluster,
    const vector<vector<double>>* z_start_per_UND_cluster,
    const vector<vector<double>>* x_end_per_UND_cluster,
    const vector<vector<double>>* y_end_per_UND_cluster,
    const vector<vector<double>>* z_end_per_UND_cluster,
    int und_fit_idx)
{
    // Calculate distances between all possible combinations of endpoints
    double d1 = calculateDistance(
        x_start_per_elec_crossing_4_alpha->at(crossing_idx).at(0),
        y_start_per_elec_crossing_4_alpha->at(crossing_idx).at(0),
        z_start_per_elec_crossing_4_alpha->at(crossing_idx).at(0),
        x_start_per_UND_cluster->at(und_idx).at(und_fit_idx),
        y_start_per_UND_cluster->at(und_idx).at(und_fit_idx),
        z_start_per_UND_cluster->at(und_idx).at(und_fit_idx)
    );

    double d2 = calculateDistance(
        x_start_per_elec_crossing_4_alpha->at(crossing_idx).at(0),
        y_start_per_elec_crossing_4_alpha->at(crossing_idx).at(0),
        z_start_per_elec_crossing_4_alpha->at(crossing_idx).at(0),
        x_end_per_UND_cluster->at(und_idx).at(und_fit_idx),
        y_end_per_UND_cluster->at(und_idx).at(und_fit_idx),
        z_end_per_UND_cluster->at(und_idx).at(und_fit_idx)
    );

    double d3 = calculateDistance(
        x_end_per_elec_crossing_4_alpha->at(crossing_idx).at(0),
        y_end_per_elec_crossing_4_alpha->at(crossing_idx).at(0),
        z_end_per_elec_crossing_4_alpha->at(crossing_idx).at(0),
        x_start_per_UND_cluster->at(und_idx).at(und_fit_idx),
        y_start_per_UND_cluster->at(und_idx).at(und_fit_idx),
        z_start_per_UND_cluster->at(und_idx).at(und_fit_idx)
    );

    double d4 = calculateDistance(
        x_end_per_elec_crossing_4_alpha->at(crossing_idx).at(0),
        y_end_per_elec_crossing_4_alpha->at(crossing_idx).at(0),
        z_end_per_elec_crossing_4_alpha->at(crossing_idx).at(0),
        x_end_per_UND_cluster->at(und_idx).at(und_fit_idx),
        y_end_per_UND_cluster->at(und_idx).at(und_fit_idx),
        z_end_per_UND_cluster->at(und_idx).at(und_fit_idx)
    );

    // Return the minimum distance
    return std::min({d1, d2, d3, d4});
}




class falaise_skeleton_module_ptd : public dpp::chain_module
{

public:
  // Constructor
  falaise_skeleton_module_ptd();

  // Destructor
  virtual ~falaise_skeleton_module_ptd();

  // Initialisation function
  virtual void initialize (const datatools::properties &,
                           datatools::service_manager &,
			   dpp::module_handle_dict_type &);

  // Event processing function
  dpp::chain_module::process_status process (datatools::things & event);
  
private:
  bool DEBUG = true;
  bool saving_file = false;
  int  ptd_event_counter;
  int event_number = 0;
  bool ptd_details, ttd_details, sd_calo_details, sd_tracker_details, pcd_details, udd_details, cd_details; 
  vector<double> energy_gamma, energy_elec, corrected_energy_elec, time_elec;
  //vector<double> vertex_start_x, vertex_start_y, vertex_start_z, vertex_end_x, vertex_end_y, vertex_end_z, vertex_gamma;

  // root file
  
  TFile *save_file;
  TTree *tree;

  // for electron clusters
  vector<vector<double>> cell_num_per_elec_cluster, anode_time_per_elec_cluster, top_cathode_per_elec_cluster, bottom_cathode_per_elec_cluster, E_OM_per_elec_cluster, z_of_cells_per_elec_cluster;
  vector<double> nb_cell_per_elec_cluster;
  vector<vector<double>> OM_num_per_elec_cluster, OM_timestamp_per_elec_cluster, OM_charge_per_elec_cluster, OM_amplitude_per_elec_cluster;
  
  vector<double> nb_of_OM_per_elec_cluster;
  
  vector<int> nb_fit_solution_per_elec_cluster;
  
  vector<vector<double>> OM_LT_only_per_elec_cluster, OM_HT_per_elec_cluster;
  
  int nb_of_elec_candidates, nb_unassociated_cells; 
  
  vector<vector<double>> x_start_per_elec_cluster, y_start_per_elec_cluster, z_start_per_elec_cluster, x_end_per_elec_cluster, y_end_per_elec_cluster, z_end_per_elec_cluster;
  

  vector<vector<double>> x_is_on_reference_source_plane, y_is_on_reference_source_plane, z_is_on_reference_source_plane;
  vector<vector<double>> x_is_on_source_foil, y_is_on_source_foil, z_is_on_source_foil;
  vector<vector<double>> x_is_on_main_calorimeter, y_is_on_main_calorimeter, z_is_on_main_calorimeter;
  vector<vector<double>> x_is_on_x_calorimeter, y_is_on_x_calorimeter, z_is_on_x_calorimeter;
  vector<vector<double>> x_is_on_gamma_veto, y_is_on_gamma_veto, z_is_on_gamma_veto;
  vector<vector<double>> x_is_on_wire, y_is_on_wire, z_is_on_wire;
  vector<vector<double>> x_is_in_gas, y_is_in_gas, z_is_in_gas;
  vector<vector<double>> x_is_on_source_gap, y_is_on_source_gap, z_is_on_source_gap;
  
  vector<vector<double>> xy_distance_from_reference_source_plane, xyz_distance_from_reference_source_plane ; 
  vector<vector<double>> xy_distance_from_source_foil, xyz_distance_from_source_foil ; 
  vector<vector<double>> xy_distance_from_main_calorimeter, xyz_distance_from_main_calorimeter ; 
  vector<vector<double>> xy_distance_from_x_calorimeter, xyz_distance_from_x_calorimeter ; 
  vector<vector<double>> xy_distance_from_gamma_veto, xyz_distance_from_gamma_veto ; 
  vector<vector<double>> xy_distance_from_wire, xyz_distance_from_wire ; 
  vector<vector<double>> xy_distance_from_gas, xyz_distance_from_gas ; 
  vector<vector<double>> xy_distance_from_source_gap, xyz_distance_from_source_gap ; 


  vector<bool> elec_cluster_is_delayed, elec_clsuter_is_prompt;

  vector<vector<double>> sigma_z_of_cells_per_elec_cluster, r_of_cells_per_elec_cluster, sigma_r_of_cells_per_elec_cluster;
  vector<int> ID_clsuter_per_elec_cluster;

bool elec_is_on_reference_source_plane = false;
bool elec_is_on_source_foil = false;
bool elec_is_on_main_calorimeter = false;
bool elec_is_on_x_calorimeter = false;
bool elec_is_on_gamma_veto = false;
bool elec_is_on_wire = false;
bool elec_is_in_gas = false;
bool elec_is_on_source_gap = false;


vector<vector<double>> vertex_is_on_reference_source_plane_per_elec_cluster; // BOOL 
vector<vector<double>> vertex_is_on_source_foil_per_elec_cluster; // BOOL 
vector<vector<double>> vertex_is_on_main_calorimeter_per_elec_cluster; // BOOL 
vector<vector<double>> vertex_is_on_x_calorimeter_per_elec_cluster; // BOOL 
vector<vector<double>> vertex_is_on_gamma_veto_per_elec_cluster; // BOOL 
vector<vector<double>> vertex_is_on_wire_per_elec_cluster; // BOOL 
vector<vector<double>> vertex_is_in_gas_per_elec_cluster; // BOOL 
vector<vector<double>> vertex_is_on_source_gap_per_elec_cluster; // BOOL 


// ELEC CROSSING : 


	int side_elec, side_UND;


	bool tracks_elec_crossing;
	int nb_of_elec_crossing;
	
	vector<vector<double>> cell_num_per_elec_crossing;
	vector<vector<double>> OM_num_per_elec_crossing;
	vector<vector<double>> E_OM_per_elec_crossing;
	vector<vector<double>> nb_of_OM_per_elec_crossing;
 
	vector<vector<double>> anode_time_per_elec_crossing;
	vector<vector<double>> top_cathode_per_elec_crossing;
	vector<vector<double>> bottom_cathode_per_elec_crossing;
	vector<vector<double>> OM_timestamp_per_elec_crossing;
	vector<vector<double>> OM_charge_per_elec_crossing;
	vector<vector<double>> OM_amplitude_per_elec_crossing;
	vector<vector<double>> OM_LT_only_per_elec_crossing;
	vector<vector<double>> OM_HT_per_elec_crossing;
	vector<vector<double>> z_of_cells_per_elec_crossing;
	vector<vector<double>> sigma_z_of_cells_per_elec_crossing;
	vector<vector<double>> r_of_cells_per_elec_crossing;
	vector<vector<double>> sigma_r_of_cells_per_elec_crossing;
	vector<vector<double>> x_start_per_elec_crossing;
	vector<vector<double>> y_start_per_elec_crossing;
	vector<vector<double>> z_start_per_elec_crossing;
	vector<vector<double>> x_end_per_elec_crossing;
	vector<vector<double>> y_end_per_elec_crossing;
	vector<vector<double>> z_end_per_elec_crossing;

	vector<vector<double>> x_is_on_reference_source_plane_per_elec_crossing;
	vector<vector<double>> y_is_on_reference_source_plane_per_elec_crossing;
	vector<vector<double>> z_is_on_reference_source_plane_per_elec_crossing;

	vector<vector<double>> x_is_on_source_foil_per_elec_crossing;
	vector<vector<double>> y_is_on_source_foil_per_elec_crossing;
	vector<vector<double>> z_is_on_source_foil_per_elec_crossing;

	vector<vector<double>> x_is_on_main_calorimeter_per_elec_crossing;
	vector<vector<double>> y_is_on_main_calorimeter_per_elec_crossing;
	vector<vector<double>> z_is_on_main_calorimeter_per_elec_crossing;

	vector<vector<double>> x_is_on_x_calorimeter_per_elec_crossing;
	vector<vector<double>> y_is_on_x_calorimeter_per_elec_crossing;
	vector<vector<double>> z_is_on_x_calorimeter_per_elec_crossing;
	
	vector<vector<double>> x_is_on_gamma_veto_per_elec_crossing;
	vector<vector<double>> y_is_on_gamma_veto_per_elec_crossing;
	vector<vector<double>> z_is_on_gamma_veto_per_elec_crossing;
	
	vector<vector<double>> x_is_on_wire_per_elec_crossing;
	vector<vector<double>> y_is_on_wire_per_elec_crossing;
	vector<vector<double>> z_is_on_wire_per_elec_crossing;
	
	vector<vector<double>> x_is_in_gas_per_elec_crossing;
	vector<vector<double>> y_is_in_gas_per_elec_crossing;
	vector<vector<double>> z_is_in_gas_per_elec_crossing;
	
	vector<vector<double>> x_is_on_source_gap_per_elec_crossing;
	vector<vector<double>> y_is_on_source_gap_per_elec_crossing;
	vector<vector<double>> z_is_on_source_gap_per_elec_crossing;

	vector<vector<double>> xy_distance_from_reference_source_plane_per_elec_crossing;
	vector<vector<double>> xy_distance_from_source_foil_per_elec_crossing;
	vector<vector<double>> xy_distance_from_main_calorimeter_per_elec_crossing;
	vector<vector<double>> xy_distance_from_x_calorimeter_per_elec_crossing;
	vector<vector<double>> xy_distance_from_gamma_veto_per_elec_crossing;
	vector<vector<double>> xy_distance_from_wire_per_elec_crossing;
	vector<vector<double>> xy_distance_from_gas_per_elec_crossing;
	vector<vector<double>> xy_distance_from_source_gap_per_elec_crossing;
	
	vector<vector<double>> xyz_distance_from_reference_source_plane_per_elec_crossing;
	vector<vector<double>> xyz_distance_from_source_foil_per_elec_crossing;
	vector<vector<double>> xyz_distance_from_main_calorimeter_per_elec_crossing;
	vector<vector<double>> xyz_distance_from_x_calorimeter_per_elec_crossing;
	vector<vector<double>> xyz_distance_from_gamma_veto_per_elec_crossing;
	vector<vector<double>> xyz_distance_from_wire_per_elec_crossing;
	vector<vector<double>> xyz_distance_from_gas_per_elec_crossing;
	vector<vector<double>> xyz_distance_from_source_gap_per_elec_crossing;

	vector<vector<double>> vertex_is_on_reference_source_plane_per_elec_crossing;
	vector<vector<double>> vertex_is_on_source_foil_per_elec_crossing;
	vector<vector<double>> vertex_is_on_main_calorimeter_per_elec_crossing;
	vector<vector<double>> vertex_is_on_x_calorimeter_per_elec_crossing;
	vector<vector<double>> vertex_is_on_gamma_veto_per_elec_crossing;
	vector<vector<double>> vertex_is_on_wire_per_elec_crossing;

	vector<vector<double>> distance_elec_UND_per_elec_crossing;
	vector<vector<double>> lenght_per_elec_crossing;
	vector<vector<double>>delta_t_cells_of_UND_per_elec_crossing;
	vector<vector<double>>delta_t_cells_of_elec_per_elec_crossing;
	vector<double> nb_of_cell_per_elec_crossing;

	vector<double> list_elec_idx_with_und_candidate_same_entry;
	vector<double> list_elec_fit_idx_with_und_candidate_same_entry;
	vector<double> list_UND_idx_with_elec_candidate_same_entry;
	vector<double> list_UND_fit_idx_with_elec_candidate_same_entry;

	int nb_tot_elec = 0;
	int nb_UND_tot = 0;

	vector<int> ID_cluster_per_elec_crossing;		// Here we are storing ID informations used to identify electron "crossing" clusters
	vector<int> ID_cluster_UND_per_elec_crossing;	// Here we are storing ID informations used to identify electron "crossing" clusters
  
	vector<double> fit_elec_index_for_each_elec_crossing;
	std::vector<bool> elec_used ; // ??
	std::vector<bool> und_used ; // ?? ICI comme ça ? 
  // for UNDEFINED CLUSTERS (electron or alpha candidates)

  
  
  vector<vector<double>> cell_num_per_UND_cluster, anode_time_per_UND_cluster, top_cathodes_per_UND_cluster, bottom_cathodes_per_UND_cluster, z_of_cell_per_UND_cluster, sigma_z_of_cells_per_UND_cluster ;
  vector<vector<double>> r_of_cells_per_UND_cluster, sigma_r_of_cells_per_UND_cluster; 
  vector<vector<double>> x_start_per_UND_cluster, y_start_per_UND_cluster, z_start_per_UND_cluster, x_end_per_UND_cluster, y_end_per_UND_cluster, z_end_per_UND_cluster;
  vector<bool> UND_cluster_is_delayed, UND_cluster_is_prompt;
  vector<int> ID_clsuter_UND;
  vector<int> nb_cell_per_UND_cluster;
  int nb_of_UND_candidates;
  vector<double> min_time_per_UND_cluster;

  vector<vector<double>> x_is_on_reference_source_plane_per_UND_cluster;
  vector<vector<double>> y_is_on_reference_source_plane_per_UND_cluster;
  vector<vector<double>> z_is_on_reference_source_plane_per_UND_cluster;
  vector<vector<double>> x_is_on_source_foil_per_UND_cluster;
  vector<vector<double>> y_is_on_source_foil_per_UND_cluster;
  vector<vector<double>> z_is_on_source_foil_per_UND_cluster;
  vector<vector<double>> x_is_on_wire_per_UND_cluster;
  vector<vector<double>> y_is_on_wire_per_UND_cluster;
  vector<vector<double>> z_is_on_wire_per_UND_cluster;
  vector<vector<double>> x_is_in_gas_per_UND_cluster;
  vector<vector<double>> y_is_in_gas_per_UND_cluster;
  vector<vector<double>> z_is_in_gas_per_UND_cluster;
  vector<vector<double>> x_is_on_source_gap_per_UND_cluster;
  vector<vector<double>> y_is_on_source_gap_per_UND_cluster;
  vector<vector<double>> z_is_on_source_gap_per_UND_cluster;
  
  vector<vector<double>> xy_distance_from_reference_source_plane_per_UND_cluster;
  vector<vector<double>> xy_distance_from_source_foil_per_UND_cluster;
  vector<vector<double>> xy_distance_from_wire_per_UND_cluster;
  vector<vector<double>> xy_distance_from_gas_per_UND_cluster;
  vector<vector<double>> xy_distance_from_source_gap_per_UND_cluster;
  vector<vector<double>> xyz_distance_from_reference_source_plane_per_UND_cluster;
  vector<vector<double>> xyz_distance_from_source_foil_per_UND_cluster;
  vector<vector<double>> xyz_distance_from_wire_per_UND_cluster;
  vector<vector<double>> xyz_distance_from_gas_per_UND_cluster;
  vector<vector<double>> xyz_distance_from_source_gap_per_UND_cluster;

  bool UND_is_on_reference_source_plane = false;
  bool UND_is_on_source_foil = false;
  bool UND_is_on_wire = false;
  bool UND_is_in_gas = false;
  bool UND_is_on_source_gap = false;

  	vector<vector<double>> vertex_is_on_reference_source_plane_per_UND_cluster; // BOOL
	vector<vector<double>> vertex_is_on_source_foil_per_UND_cluster; // BOOL
	vector<vector<double>> vertex_is_on_wire_per_UND_cluster; // BOOL
	vector<vector<double>> vertex_is_in_gas_per_UND_cluster; // BOOL
	vector<vector<double>> vertex_is_on_source_gap_per_UND_cluster; // BOOL

	vector<int>nb_fit_solution_per_UND_cluster;
  	   
  
  // OM Isolated (gamma) part : 
  
  int nb_isolated_calo;
  vector<double> E_isolated_calo,sigma_E_isolated_calo, time_isolated_calo, sigma_time_isolated_calo, vertex_isolated_calo, isolated_calo_num; // for PTD
  vector<double> isolated_calo_timestamp, isolated_calo_charge, isolated_calo_amplitude,sigma_isolated_calo_timestamp, sigma_isolated_calo_charge, sigma_isolated_calo_amplitude ; // for pCD
  vector<char> isolated_calo_type; // PTD
  vector<bool> isolated_calo_low_threshold_only, isolated_calo_high_threshold;
  // unassociated cells
  
  vector<int> unassociated_cells_num;
  vector<double> anode_unassociated_cells, top_cathodes_of_unassociated_cells, bottom_cathodes_of_unassociated_cells, z_of_unassociated_cells; 
  
  // Tag
  bool evt_isolated_calo;
  bool tracks_with_associated_calo;
  bool tracks_without_associated_calo;
  
  // Macro to register the module
  DPP_MODULE_REGISTRATION_INTERFACE(falaise_skeleton_module_ptd);

};

////////////////////////////////////////////////////////////////////

// Macro to add the module in the global register of data processing modules:
// The module defined by this class 'falaise_skeleton_module_ptd' will be registered
// with the label ID 'FalaiseSkeletonModule_PTD' (to use in pipeline configuration file)
DPP_MODULE_REGISTRATION_IMPLEMENT(falaise_skeleton_module_ptd, "FalaiseSkeletonModule_PTD")


falaise_skeleton_module_ptd::falaise_skeleton_module_ptd()
{
  std::cout << "falaise_skeleton_module_ptd::falaise_skeleton_module_ptd() called" << std::endl;
  if(saving_file){
  	save_file = new TFile("ptd_1166_WITH_ELEC_UND_AMBIGUITY_CHECKED.root", "RECREATE");
  	tree = new TTree("particules", "PTD data informations extracted ==> for second analysis");

  
  	tree->Branch("event_number", &event_number);
  	// Isolated calo part
  	tree->Branch("nb_isolated_calo",                  &nb_isolated_calo);
  	tree->Branch("E_isolated_calo",                   &E_isolated_calo);
  	tree->Branch("time_isolated_calo",                &time_isolated_calo);
  	tree->Branch("sigma_E_isolated_calo",             &sigma_E_isolated_calo);
  	tree->Branch("sigma_time_isolated_calo",          &sigma_time_isolated_calo);
  	tree->Branch("isolated_calo_type",                &isolated_calo_type);
  	tree->Branch("isolated_calo_timestamp",           &isolated_calo_timestamp);
  	tree->Branch("isolated_calo_charge",              &isolated_calo_charge);
  	tree->Branch("isolated_calo_amplitude",           &isolated_calo_amplitude); 
  	tree->Branch("isolated_calo_num",                 &isolated_calo_num);
  	tree->Branch("sigma_isolated_calo_timestamp",     &sigma_isolated_calo_timestamp);
  	tree->Branch("sigma_isolated_calo_charge",        &sigma_isolated_calo_charge);
  	tree->Branch("sigma_isolated_calo_amplitude",     &sigma_isolated_calo_amplitude);
  	tree->Branch("evt_isolated_calo",                 &evt_isolated_calo);
  	tree->Branch("isolated_calo_low_threshold_only",  &isolated_calo_low_threshold_only);
  	tree->Branch("isolated_calo_high_threshold",      &isolated_calo_high_threshold);




  	//tree->Branch("", &);

  	// Electron candidate
  	tree->Branch("tracks_with_associated_calo", &tracks_with_associated_calo);
  	tree->Branch("nb_of_elec_candidates", &nb_of_elec_candidates);
  	tree->Branch("cell_num_per_elec_cluster", &cell_num_per_elec_cluster);
  	tree->Branch("OM_num_per_elec_cluster", &OM_num_per_elec_cluster);
  	tree->Branch("E_OM_per_elec_cluster", &E_OM_per_elec_cluster);
  	tree->Branch("nb_of_OM_per_elec_cluster", &nb_of_OM_per_elec_cluster);
  	tree->Branch("anode_time_per_elec_cluster", &anode_time_per_elec_cluster);
  	tree->Branch("top_cathode_per_elec_cluster", &top_cathode_per_elec_cluster);
  	tree->Branch("bottom_cathode_per_elec_cluster", &bottom_cathode_per_elec_cluster);
  	tree->Branch("OM_timestamp_per_elec_cluster", &OM_timestamp_per_elec_cluster);
  	tree->Branch("OM_charge_per_elec_cluster", &OM_charge_per_elec_cluster);
  	tree->Branch("OM_amplitude_per_elec_cluster", &OM_amplitude_per_elec_cluster);
  	tree->Branch("OM_LT_only_per_elec_cluster", &OM_LT_only_per_elec_cluster);
  	tree->Branch("OM_HT_per_elec_cluster", &OM_HT_per_elec_cluster);
  	tree->Branch("z_of_cells_per_elec_cluster", &z_of_cells_per_elec_cluster);
  	tree->Branch("sigma_z_of_cells_per_elec_cluster", &sigma_z_of_cells_per_elec_cluster);
  	tree->Branch("r_of_cells_per_elec_cluster", &r_of_cells_per_elec_cluster);
  	tree->Branch("sigma_r_of_cells_per_elec_cluster", &sigma_r_of_cells_per_elec_cluster);
  	tree->Branch("elec_cluster_is_delayed", &elec_cluster_is_delayed);
  	tree->Branch("elec_clsuter_is_prompt", &elec_clsuter_is_prompt);
  	tree->Branch("x_start_per_elec_cluster", &x_start_per_elec_cluster);
  	tree->Branch("y_start_per_elec_cluster", &y_start_per_elec_cluster);
  	tree->Branch("z_start_per_elec_cluster", &z_start_per_elec_cluster);
  	tree->Branch("x_end_per_elec_cluster", &x_end_per_elec_cluster);
  	tree->Branch("y_end_per_elec_cluster", &y_end_per_elec_cluster);
  	tree->Branch("z_end_per_elec_cluster", &z_end_per_elec_cluster);
  	tree->Branch("ID_clsuter_per_elec_cluster", &ID_clsuter_per_elec_cluster);
  	tree->Branch("nb_fit_solution_per_elec_cluster", &nb_fit_solution_per_elec_cluster);
  	tree->Branch("nb_cell_per_elec_cluster", &nb_cell_per_elec_cluster);
	
	tree->Branch("x_is_on_reference_source_plane", &x_is_on_reference_source_plane);
	tree->Branch("y_is_on_reference_source_plane", &y_is_on_reference_source_plane);
	tree->Branch("z_is_on_reference_source_plane", &z_is_on_reference_source_plane);
	tree->Branch("x_is_on_source_foil", &x_is_on_source_foil);
	tree->Branch("y_is_on_source_foil", &y_is_on_source_foil);
	tree->Branch("z_is_on_source_foil", &z_is_on_source_foil);
	tree->Branch("x_is_on_main_calorimeter", &x_is_on_main_calorimeter);
	tree->Branch("y_is_on_main_calorimeter", &y_is_on_main_calorimeter);
	tree->Branch("z_is_on_main_calorimeter", &z_is_on_main_calorimeter);
	tree->Branch("x_is_on_x_calorimeter", &x_is_on_x_calorimeter);
	tree->Branch("y_is_on_x_calorimeter", &y_is_on_x_calorimeter);
	tree->Branch("z_is_on_x_calorimeter", &z_is_on_x_calorimeter);
	tree->Branch("x_is_on_gamma_veto", &x_is_on_gamma_veto);
	tree->Branch("y_is_on_gamma_veto", &y_is_on_gamma_veto);
	tree->Branch("z_is_on_gamma_veto", &z_is_on_gamma_veto);
	tree->Branch("x_is_on_wire", &x_is_on_wire);
	tree->Branch("y_is_on_wire", &y_is_on_wire);
	tree->Branch("z_is_on_wire", &z_is_on_wire);
	tree->Branch("x_is_in_gas", &x_is_in_gas);
	tree->Branch("y_is_in_gas", &y_is_in_gas);
	tree->Branch("z_is_in_gas", &z_is_in_gas);
	tree->Branch("x_is_on_source_gap", &x_is_on_source_gap);
	tree->Branch("y_is_on_source_gap", &y_is_on_source_gap);
	tree->Branch("z_is_on_source_gap", &z_is_on_source_gap);
	  
	tree->Branch("xy_distance_from_reference_source_plane", &xy_distance_from_reference_source_plane);
	tree->Branch("xy_distance_from_source_foil", &xy_distance_from_source_foil);
	tree->Branch("xy_distance_from_main_calorimeter", &xy_distance_from_main_calorimeter);
	tree->Branch("xy_distance_from_x_calorimeter", &xy_distance_from_x_calorimeter);
	tree->Branch("xy_distance_from_gamma_veto", &xy_distance_from_gamma_veto);
	tree->Branch("xy_distance_from_wire", &xy_distance_from_wire);
	tree->Branch("xy_distance_from_gas", &xy_distance_from_gas);
	tree->Branch("xy_distance_from_source_gap", &xy_distance_from_source_gap);
	  
	tree->Branch("xyz_distance_from_reference_source_plane", &xyz_distance_from_reference_source_plane);
	tree->Branch("xyz_distance_from_source_foil", &xyz_distance_from_source_foil);
	tree->Branch("xyz_distance_from_main_calorimeter", &xyz_distance_from_main_calorimeter);
	tree->Branch("xyz_distance_from_x_calorimeter", &xyz_distance_from_x_calorimeter);
	tree->Branch("xyz_distance_from_gamma_veto", &xyz_distance_from_gamma_veto);
	tree->Branch("xyz_distance_from_wire", &xyz_distance_from_wire);
	tree->Branch("xyz_distance_from_gas", &xyz_distance_from_gas);
	tree->Branch("xyz_distance_from_source_gap", &xyz_distance_from_source_gap);

	tree->Branch("elec_is_on_reference_source_plane", &elec_is_on_reference_source_plane);
	tree->Branch("elec_is_on_source_foil", &elec_is_on_source_foil);
	tree->Branch("elec_is_on_main_calorimeter", &elec_is_on_main_calorimeter);
	tree->Branch("elec_is_on_x_calorimeter", &elec_is_on_x_calorimeter);
	tree->Branch("elec_is_on_gamma_veto", &elec_is_on_gamma_veto);
	tree->Branch("elec_is_on_wire", &elec_is_on_wire);
	tree->Branch("elec_is_in_gas", &elec_is_in_gas);
	tree->Branch("elec_is_on_source_gap", &elec_is_on_source_gap);

	tree->Branch("vertex_is_on_reference_source_plane_per_elec_cluster", &vertex_is_on_reference_source_plane_per_elec_cluster);
	tree->Branch("vertex_is_on_source_foil_per_elec_cluster", &vertex_is_on_source_foil_per_elec_cluster);
	tree->Branch("vertex_is_on_main_calorimeter_per_elec_cluster", &vertex_is_on_main_calorimeter_per_elec_cluster);
	tree->Branch("vertex_is_on_x_calorimeter_per_elec_cluster", &vertex_is_on_x_calorimeter_per_elec_cluster);
	tree->Branch("vertex_is_on_gamma_veto_per_elec_cluster", &vertex_is_on_gamma_veto_per_elec_cluster);
	tree->Branch("vertex_is_on_wire_per_elec_cluster", &vertex_is_on_wire_per_elec_cluster);
	tree->Branch("vertex_is_in_gas_per_elec_cluster", &vertex_is_in_gas_per_elec_cluster);
	tree->Branch("vertex_is_on_source_gap_per_elec_cluster", &vertex_is_on_source_gap_per_elec_cluster);
  

  	//UNDIFINED TRACKS : 
  	tree->Branch("tracks_without_associated_calo", &tracks_without_associated_calo);
  	tree->Branch("nb_of_UND_candidates", &nb_of_UND_candidates);
  	tree->Branch("cell_num_per_UND_cluster", &cell_num_per_UND_cluster);
  	tree->Branch("anode_time_per_UND_cluster", &anode_time_per_UND_cluster);
  	tree->Branch("top_cathodes_per_UND_cluster", &top_cathodes_per_UND_cluster);
  	tree->Branch("bottom_cathodes_per_UND_cluster", &bottom_cathodes_per_UND_cluster);
  	tree->Branch("z_of_cell_per_UND_cluster", &z_of_cell_per_UND_cluster);
  	tree->Branch("sigma_z_of_cells_per_UND_cluster", &sigma_z_of_cells_per_UND_cluster);
  	tree->Branch("r_of_cells_per_UND_cluster", &r_of_cells_per_UND_cluster);
  	tree->Branch("sigma_r_of_cells_per_UND_cluster", &sigma_r_of_cells_per_UND_cluster);
  	tree->Branch("UND_cluster_is_delayed", &UND_cluster_is_delayed);
  	tree->Branch("UND_cluster_is_prompt", &UND_cluster_is_prompt);
  	tree->Branch("ID_clsuter_UND", &ID_clsuter_UND);
  	tree->Branch("x_start_per_UND_cluster", &x_start_per_UND_cluster);
  	tree->Branch("y_start_per_UND_cluster", &y_start_per_UND_cluster);
  	tree->Branch("z_start_per_UND_cluster", &z_start_per_UND_cluster);
  	tree->Branch("x_end_per_UND_cluster", &x_end_per_UND_cluster);
  	tree->Branch("y_end_per_UND_cluster", &y_end_per_UND_cluster);
  	tree->Branch("z_end_per_UND_cluster", &z_end_per_UND_cluster);
  	tree->Branch("nb_cell_per_UND_cluster", &nb_cell_per_UND_cluster);
	tree->Branch("min_time_per_UND_cluster", &min_time_per_UND_cluster);
	tree->Branch("nb_fit_solution_per_UND_cluster", &nb_fit_solution_per_UND_cluster);
	tree->Branch("x_is_on_reference_source_plane_per_UND_cluster", &x_is_on_reference_source_plane_per_UND_cluster);
	tree->Branch("y_is_on_reference_source_plane_per_UND_cluster", &y_is_on_reference_source_plane_per_UND_cluster);
	tree->Branch("z_is_on_reference_source_plane_per_UND_cluster", &z_is_on_reference_source_plane_per_UND_cluster);
	tree->Branch("x_is_on_source_foil_per_UND_cluster", &x_is_on_source_foil_per_UND_cluster);
	tree->Branch("y_is_on_source_foil_per_UND_cluster", &y_is_on_source_foil_per_UND_cluster);
	tree->Branch("z_is_on_source_foil_per_UND_cluster", &z_is_on_source_foil_per_UND_cluster);
	tree->Branch("x_is_on_wire_per_UND_cluster", &x_is_on_wire_per_UND_cluster);
	tree->Branch("y_is_on_wire_per_UND_cluster", &y_is_on_wire_per_UND_cluster);
	tree->Branch("z_is_on_wire_per_UND_cluster", &z_is_on_wire_per_UND_cluster);
	tree->Branch("x_is_in_gas_per_UND_cluster", &x_is_in_gas_per_UND_cluster);
	tree->Branch("y_is_in_gas_per_UND_cluster", &y_is_in_gas_per_UND_cluster);
	tree->Branch("z_is_in_gas_per_UND_cluster", &z_is_in_gas_per_UND_cluster);
	tree->Branch("x_is_on_source_gap_per_UND_cluster", &x_is_on_source_gap_per_UND_cluster);
	tree->Branch("y_is_on_source_gap_per_UND_cluster", &y_is_on_source_gap_per_UND_cluster);
	tree->Branch("z_is_on_source_gap_per_UND_cluster", &z_is_on_source_gap_per_UND_cluster);
	tree->Branch("xy_distance_from_reference_source_plane_per_UND_cluster", &xy_distance_from_reference_source_plane_per_UND_cluster);
	tree->Branch("xy_distance_from_source_foil_per_UND_cluster", &xy_distance_from_source_foil_per_UND_cluster);
	tree->Branch("xy_distance_from_wire_per_UND_cluster", &xy_distance_from_wire_per_UND_cluster);
	tree->Branch("xy_distance_from_gas_per_UND_cluster", &xy_distance_from_gas_per_UND_cluster);
	tree->Branch("xy_distance_from_source_gap_per_UND_cluster", &xy_distance_from_source_gap_per_UND_cluster);
	tree->Branch("xyz_distance_from_reference_source_plane_per_UND_cluster", &xyz_distance_from_reference_source_plane_per_UND_cluster);
	tree->Branch("xyz_distance_from_source_foil_per_UND_cluster", &xyz_distance_from_source_foil_per_UND_cluster);
	tree->Branch("xyz_distance_from_wire_per_UND_cluster", &xyz_distance_from_wire_per_UND_cluster);
	tree->Branch("xyz_distance_from_gas_per_UND_cluster", &xyz_distance_from_gas_per_UND_cluster);
	tree->Branch("xyz_distance_from_source_gap_per_UND_cluster", &xyz_distance_from_source_gap_per_UND_cluster);
	tree->Branch("UND_is_on_reference_source_plane", &UND_is_on_reference_source_plane);
	tree->Branch("UND_is_on_source_foil", &UND_is_on_source_foil);
	tree->Branch("UND_is_on_wire", &UND_is_on_wire);
	tree->Branch("UND_is_in_gas", &UND_is_in_gas);
	tree->Branch("UND_is_on_source_gap", &UND_is_on_source_gap);
	tree->Branch("vertex_is_on_reference_source_plane_per_UND_cluster", &vertex_is_on_reference_source_plane_per_UND_cluster);
	tree->Branch("vertex_is_on_source_foil_per_UND_cluster", &vertex_is_on_source_foil_per_UND_cluster);
	tree->Branch("vertex_is_on_wire_per_UND_cluster", &vertex_is_on_wire_per_UND_cluster);
	tree->Branch("vertex_is_in_gas_per_UND_cluster", &vertex_is_in_gas_per_UND_cluster);
	tree->Branch("vertex_is_on_source_gap_per_UND_cluster", &vertex_is_on_source_gap_per_UND_cluster);

	// ELEC CROSSING: 



	
  }

}


falaise_skeleton_module_ptd::~falaise_skeleton_module_ptd()
{
	if(saving_file){
		save_file->cd();
		tree->Write();
  		save_file->Close();
		delete save_file;

	}
  
  
  
  std::cout << "falaise_skeleton_module_ptd::~falaise_skeleton_module_ptd() called" << std::endl;
}


void falaise_skeleton_module_ptd::initialize (const datatools::properties & module_properties, datatools::service_manager &, dpp::module_handle_dict_type &)
{
	std::cout << "falaise_skeleton_module_ptd::initialize() called" << std::endl;

	ptd_event_counter = 0;

	if ( module_properties.has_key("ptd_details"))
    	ptd_details = module_properties.fetch_boolean("ptd_details");
  	else ptd_details = false;

  	if (module_properties.has_key("pcd_details"))
    	pcd_details= module_properties.fetch_boolean("pcd_details");
  	else pcd_details = false;

  	if(module_properties.has_key("ttd_details"))
    	ttd_details = module_properties.fetch_boolean("ttd_details");
  	else ttd_details = false;

  	if( module_properties.has_key("calo_details"))
    	sd_calo_details = module_properties.fetch_boolean("calo_details");
  	else sd_calo_details = false;

	  if(module_properties.has_key("tracker_details"))
	    sd_tracker_details = module_properties.fetch_boolean("tracker_details");
	  else sd_tracker_details = false;

	  if(module_properties.has_key("udd_details"))
	    udd_details = module_properties.fetch_boolean("udd_details");
	  else udd_details = false;

	  if(module_properties.has_key("cd_details"))
	  	cd_details = module_properties.fetch_boolean("cd_details");
	  else cd_details = false;

  
  	this->_set_initialized(true);
}

dpp::chain_module::process_status falaise_skeleton_module_ptd::process (datatools::things & event)
{
  	// Skip processing if PTD bank is not present
  	if (!event.has("PTD")){
    	std::cout << "======== no PTD bank in event " << ptd_event_counter++ << " ========" << std::endl;
      	return dpp::base_module::PROCESS_SUCCESS;
      	event_number++;
    	}
  	if(!event.has("pCD")){
    	//std::cout << "======== no pCD bank in event " << ptd_event_counter++ << " ========" << std::endl;
    	return dpp::base_module::PROCESS_SUCCESS;
  	}
  	if(!event.has("UDD")){
    	return dpp::base_module::PROCESS_SUCCESS;
  	}
	if(!event.has("CD")){
		return dpp::base_module::PROCESS_SUCCESS;
	}
	if(!event.has("TTD")){
		return dpp::base_module::PROCESS_SUCCESS;
	}
  
  	// Retrieve the PTD bank  
  	const snemo::datamodel::particle_track_data & PTD = event.get<snemo::datamodel::particle_track_data>("PTD");
  	
	if(saving_file==false){
		std::cout <<"\n" <<"================ PTD bank of event " << ptd_event_counter++ << " ==================" << std::endl;
	}
	
  	// Brows>e calibrated calorimeter hits
 
	//std::cout << "=> " << PTD.particles().size() << " particle(s)" << std::endl;

  	const snemo::datamodel::precalibrated_data & PCD = event.get<snemo::datamodel::precalibrated_data>("pCD");
  	//std::cout << "======== PCD bank of event " << ptd_event_counter++ <<  " ========" << std::endl;

  	const snemo::datamodel::unified_digitized_data & UDD = event.get<snemo::datamodel::unified_digitized_data>("UDD");

	const snemo::datamodel::tracker_trajectory_data & TTD = event.get<snemo::datamodel::tracker_trajectory_data>("TTD");

  	// Enregistrer les électrons
  	// Enregistrer les traces sans OMs
  	// Faire tag avec une ou plusieurs solutions, plusieurs OMs ou non
  	// Trouver comment enregistrer les gammas ? Trouvé : PTD.hasIsolatedCalorimeters() ? PTD->isolatedCalorimeters()


  	// TAG

  	evt_isolated_calo = false;
  	tracks_with_associated_calo = false;
  	tracks_without_associated_calo= false;
  
  	// ISOLATED CALO 
  	nb_isolated_calo = 0;
  	E_isolated_calo.clear();
  	time_isolated_calo.clear();
  	sigma_E_isolated_calo.clear();
  	sigma_time_isolated_calo.clear();
  	isolated_calo_type.clear();
  	isolated_calo_timestamp.clear();
  	isolated_calo_charge.clear();
  	isolated_calo_amplitude.clear();
  	isolated_calo_num.clear();
  	sigma_isolated_calo_timestamp.clear();
  	sigma_isolated_calo_charge.clear();
  	sigma_isolated_calo_amplitude.clear();
  	isolated_calo_low_threshold_only.clear();
  	isolated_calo_high_threshold.clear();

  	// ELEC CANDIDATE
  	cell_num_per_elec_cluster.clear();
  	OM_num_per_elec_cluster.clear();
  	E_OM_per_elec_cluster.clear();
  	nb_of_OM_per_elec_cluster.clear();
  	anode_time_per_elec_cluster.clear();
  	top_cathode_per_elec_cluster.clear();
  	bottom_cathode_per_elec_cluster.clear();
	OM_timestamp_per_elec_cluster.clear();
	OM_charge_per_elec_cluster.clear();
	OM_amplitude_per_elec_cluster.clear();
	OM_LT_only_per_elec_cluster.clear();
	OM_HT_per_elec_cluster.clear();
	z_of_cells_per_elec_cluster.clear();
	sigma_z_of_cells_per_elec_cluster.clear();
	r_of_cells_per_elec_cluster.clear();
	sigma_r_of_cells_per_elec_cluster.clear();
	elec_cluster_is_delayed.clear();
	elec_clsuter_is_prompt.clear();
	x_start_per_elec_cluster.clear();
	y_start_per_elec_cluster.clear();
	z_start_per_elec_cluster.clear();
	x_end_per_elec_cluster.clear();
	y_end_per_elec_cluster.clear();
	z_end_per_elec_cluster.clear();
	ID_clsuter_per_elec_cluster.clear();
	nb_fit_solution_per_elec_cluster.clear();
	nb_of_elec_candidates = 0;
	nb_cell_per_elec_cluster.clear();


	x_is_on_reference_source_plane.clear();
	y_is_on_reference_source_plane.clear();
	z_is_on_reference_source_plane.clear();
	x_is_on_source_foil.clear();
	y_is_on_source_foil.clear();
	z_is_on_source_foil.clear();
	x_is_on_main_calorimeter.clear();
	y_is_on_main_calorimeter.clear();
	z_is_on_main_calorimeter.clear();
	x_is_on_x_calorimeter.clear();
	y_is_on_x_calorimeter.clear();
	z_is_on_x_calorimeter.clear();
	x_is_on_gamma_veto.clear();
	y_is_on_gamma_veto.clear();
	z_is_on_gamma_veto.clear();
	x_is_on_wire.clear();
	y_is_on_wire.clear();
	z_is_on_wire.clear();
	x_is_in_gas.clear();
	y_is_in_gas.clear();
	z_is_in_gas.clear();
	x_is_on_source_gap.clear();
	y_is_on_source_gap.clear();
	z_is_on_source_gap.clear();

	xy_distance_from_reference_source_plane.clear();
	xy_distance_from_source_foil.clear();
	xy_distance_from_main_calorimeter.clear();
	xy_distance_from_x_calorimeter.clear();
	xy_distance_from_gamma_veto.clear();
	xy_distance_from_wire.clear();
	xy_distance_from_gas.clear();
	xy_distance_from_source_gap.clear();

	xyz_distance_from_reference_source_plane.clear();
	xyz_distance_from_source_foil.clear();
	xyz_distance_from_main_calorimeter.clear();
	xyz_distance_from_x_calorimeter.clear();
	xyz_distance_from_gamma_veto.clear();
	xyz_distance_from_wire.clear();
	xyz_distance_from_gas.clear();
	xyz_distance_from_source_gap.clear();


	elec_is_on_reference_source_plane = false;
	elec_is_on_source_foil = false;
	elec_is_on_main_calorimeter = false;
	elec_is_on_x_calorimeter = false;
	elec_is_on_gamma_veto = false;
	elec_is_on_wire = false;
	elec_is_in_gas = false;
	elec_is_on_source_gap = false;




	vertex_is_on_reference_source_plane_per_elec_cluster.clear();
	vertex_is_on_source_foil_per_elec_cluster.clear();
	vertex_is_on_main_calorimeter_per_elec_cluster.clear();
	vertex_is_on_x_calorimeter_per_elec_cluster.clear();
	vertex_is_on_gamma_veto_per_elec_cluster.clear();
	vertex_is_on_wire_per_elec_cluster.clear();
	vertex_is_in_gas_per_elec_cluster.clear();
	vertex_is_on_source_gap_per_elec_cluster.clear();

	// UNDEFINED CLUSTER (Tracks without OM associated)
	
	nb_of_UND_candidates = 0;
	cell_num_per_UND_cluster.clear();
	anode_time_per_UND_cluster.clear();
	top_cathodes_per_UND_cluster.clear();
	bottom_cathodes_per_UND_cluster.clear();
	z_of_cell_per_UND_cluster.clear(); 
	sigma_z_of_cells_per_UND_cluster.clear();
	r_of_cells_per_UND_cluster.clear();
	sigma_r_of_cells_per_UND_cluster.clear();
	UND_cluster_is_delayed.clear();
	UND_cluster_is_prompt.clear();
	ID_clsuter_UND.clear();
	x_start_per_UND_cluster.clear();
    y_start_per_UND_cluster.clear();
    z_start_per_UND_cluster.clear();
    x_end_per_UND_cluster.clear();
    y_end_per_UND_cluster.clear();
    z_end_per_UND_cluster.clear();
	nb_cell_per_UND_cluster.clear();
	min_time_per_UND_cluster.clear();
	nb_fit_solution_per_UND_cluster.clear();

	x_is_on_reference_source_plane_per_UND_cluster.clear();
	y_is_on_reference_source_plane_per_UND_cluster.clear();
	z_is_on_reference_source_plane_per_UND_cluster.clear();
	x_is_on_source_foil_per_UND_cluster.clear();
	y_is_on_source_foil_per_UND_cluster.clear();
	z_is_on_source_foil_per_UND_cluster.clear();
	x_is_on_wire_per_UND_cluster.clear();
	y_is_on_wire_per_UND_cluster.clear();
	z_is_on_wire_per_UND_cluster.clear();
	x_is_in_gas_per_UND_cluster.clear();
	y_is_in_gas_per_UND_cluster.clear();
	z_is_in_gas_per_UND_cluster.clear();
	x_is_on_source_gap_per_UND_cluster.clear();
	y_is_on_source_gap_per_UND_cluster.clear();
	z_is_on_source_gap_per_UND_cluster.clear();
	xy_distance_from_reference_source_plane_per_UND_cluster.clear();
	xy_distance_from_source_foil_per_UND_cluster.clear();
	xy_distance_from_wire_per_UND_cluster.clear();
	xy_distance_from_gas_per_UND_cluster.clear();
	xy_distance_from_source_gap_per_UND_cluster.clear();
	xyz_distance_from_reference_source_plane_per_UND_cluster.clear();
	xyz_distance_from_source_foil_per_UND_cluster.clear();
	xyz_distance_from_wire_per_UND_cluster.clear();
	xyz_distance_from_gas_per_UND_cluster.clear();
	xyz_distance_from_source_gap_per_UND_cluster.clear();

	UND_is_on_reference_source_plane = false;
	UND_is_on_source_foil = false;
	UND_is_on_wire = false;
	UND_is_in_gas = false;
	UND_is_on_source_gap = false;



	vertex_is_on_reference_source_plane_per_UND_cluster.clear();
	vertex_is_on_source_foil_per_UND_cluster.clear();
	vertex_is_on_wire_per_UND_cluster.clear();
	vertex_is_in_gas_per_UND_cluster.clear();
	vertex_is_on_source_gap_per_UND_cluster.clear();

	// ELEC CROSSING : 
	tracks_elec_crossing = false;
		nb_of_elec_crossing = 0;
		cell_num_per_elec_crossing.clear();
		OM_num_per_elec_crossing.clear();
		E_OM_per_elec_crossing.clear();
		nb_of_OM_per_elec_crossing.clear();
		anode_time_per_elec_crossing.clear();
		top_cathode_per_elec_crossing.clear();
		bottom_cathode_per_elec_crossing.clear();
		OM_timestamp_per_elec_crossing.clear();
		OM_charge_per_elec_crossing.clear();
		OM_amplitude_per_elec_crossing.clear();
		OM_LT_only_per_elec_crossing.clear();
		OM_HT_per_elec_crossing.clear();
		z_of_cells_per_elec_crossing.clear();
		sigma_z_of_cells_per_elec_crossing.clear();
		r_of_cells_per_elec_crossing.clear();
		sigma_r_of_cells_per_elec_crossing.clear();
		x_start_per_elec_crossing.clear();
		y_start_per_elec_crossing.clear();
		z_start_per_elec_crossing.clear();
		x_end_per_elec_crossing.clear();
		y_end_per_elec_crossing.clear();
		z_end_per_elec_crossing.clear();
		x_is_on_reference_source_plane_per_elec_crossing.clear();
		y_is_on_reference_source_plane_per_elec_crossing.clear();
		z_is_on_reference_source_plane_per_elec_crossing.clear();
		x_is_on_source_foil_per_elec_crossing.clear();
		y_is_on_source_foil_per_elec_crossing.clear();
		z_is_on_source_foil_per_elec_crossing.clear();
		x_is_on_main_calorimeter_per_elec_crossing.clear();
		y_is_on_main_calorimeter_per_elec_crossing.clear();
		z_is_on_main_calorimeter_per_elec_crossing.clear();
		x_is_on_x_calorimeter_per_elec_crossing.clear();
		y_is_on_x_calorimeter_per_elec_crossing.clear();
		z_is_on_x_calorimeter_per_elec_crossing.clear();
		x_is_on_gamma_veto_per_elec_crossing.clear();
		y_is_on_gamma_veto_per_elec_crossing.clear();
		z_is_on_gamma_veto_per_elec_crossing.clear();
		x_is_on_wire_per_elec_crossing.clear();
		y_is_on_wire_per_elec_crossing.clear();
		z_is_on_wire_per_elec_crossing.clear();
		x_is_in_gas_per_elec_crossing.clear();
		y_is_in_gas_per_elec_crossing.clear();
		z_is_in_gas_per_elec_crossing.clear();
		x_is_on_source_gap_per_elec_crossing.clear();
		y_is_on_source_gap_per_elec_crossing.clear();
		z_is_on_source_gap_per_elec_crossing.clear();
		xy_distance_from_reference_source_plane_per_elec_crossing.clear();
		xy_distance_from_source_foil_per_elec_crossing.clear();
		xy_distance_from_main_calorimeter_per_elec_crossing.clear();
		xy_distance_from_x_calorimeter_per_elec_crossing.clear();
		xy_distance_from_gamma_veto_per_elec_crossing.clear();
		xy_distance_from_wire_per_elec_crossing.clear();
		xy_distance_from_gas_per_elec_crossing.clear();
		xy_distance_from_source_gap_per_elec_crossing.clear();
		xyz_distance_from_reference_source_plane_per_elec_crossing.clear();
		xyz_distance_from_source_foil_per_elec_crossing.clear();
		xyz_distance_from_main_calorimeter_per_elec_crossing.clear();
		xyz_distance_from_x_calorimeter_per_elec_crossing.clear();
		xyz_distance_from_gamma_veto_per_elec_crossing.clear();
		xyz_distance_from_wire_per_elec_crossing.clear();
		xyz_distance_from_gas_per_elec_crossing.clear();
		xyz_distance_from_source_gap_per_elec_crossing.clear();
		vertex_is_on_reference_source_plane_per_elec_crossing.clear();
		vertex_is_on_source_foil_per_elec_crossing.clear();
		vertex_is_on_main_calorimeter_per_elec_crossing.clear();
		vertex_is_on_x_calorimeter_per_elec_crossing.clear();
		vertex_is_on_gamma_veto_per_elec_crossing.clear();
		vertex_is_on_wire_per_elec_crossing.clear();
		distance_elec_UND_per_elec_crossing.clear();
		lenght_per_elec_crossing.clear();
		delta_t_cells_of_UND_per_elec_crossing.clear();
		delta_t_cells_of_elec_per_elec_crossing.clear();
		nb_tot_elec = 0;
		nb_UND_tot = 0;
		
		nb_of_cell_per_elec_crossing.clear();
		
		list_elec_idx_with_und_candidate_same_entry.clear();
		list_elec_fit_idx_with_und_candidate_same_entry.clear();
		list_UND_idx_with_elec_candidate_same_entry.clear();
		list_UND_fit_idx_with_elec_candidate_same_entry.clear();
		fit_elec_index_for_each_elec_crossing.clear();
		ID_cluster_per_elec_crossing.clear();
		ID_cluster_UND_per_elec_crossing.clear();
		elec_used.clear();
		und_used.clear();
	

	//std::cout<<"nb particles in the current event : "<<PTD.particles().size()<<std::endl;
	if (ptd_details)
    {
      
    	// Gammas

    	if(PTD.hasIsolatedCalorimeters()){
			const snemo::datamodel::CalorimeterHitHdlCollection & Calo_hit_Hdl = PTD.isolatedCalorimeters();
			const auto pCD_calo_isolated = PCD.calorimeter_hits();
			// Data from PTD bank
			for(const auto & OM_hit_isolated : Calo_hit_Hdl){
			  	nb_isolated_calo++;
			  	E_isolated_calo.push_back(OM_hit_isolated->get_energy());
			  	time_isolated_calo.push_back(OM_hit_isolated->get_time()/CLHEP::second); 
			  	int isolated_OM_num = snemo::datamodel::om_num(OM_hit_isolated->get_geom_id());
			  	isolated_calo_num.push_back(isolated_OM_num);
			  	// Uncertainties :

			  	sigma_E_isolated_calo.push_back(OM_hit_isolated->get_sigma_energy());
			  	sigma_time_isolated_calo.push_back(OM_hit_isolated->get_sigma_time()/CLHEP::second);

			  	// type & debug

			  	string OM_label = snemo::datamodel::om_label(OM_hit_isolated->get_geom_id());
			  	isolated_calo_type.push_back(OM_label[0]);
			  	//std::cout<<"OM LABEL FROM STRING = "<< OM_label[0]<<std::endl; // ==> Return string "M", "X" or "G" for Main, X or Gamma Veto to determine the OM type  

			}
			// Add Data from pCD bank
			for(const auto pCD_calo : pCD_calo_isolated){
		  		// Data from pCD bank
		  		if(std::find(isolated_calo_num.begin(), isolated_calo_num.end(), snemo::datamodel::om_num(pCD_calo->get_geom_id())) != isolated_calo_num.end()){
		    		// check if the current pCD calo is already selected previously in PTD bank analysis 
		    		isolated_calo_timestamp.push_back(pCD_calo->get_time()/CLHEP::second);
		    		isolated_calo_charge.push_back(pCD_calo->get_charge());
		    		isolated_calo_amplitude.push_back(pCD_calo->get_amplitude());

		    		// Uncertainties :
		    		sigma_isolated_calo_timestamp.push_back(pCD_calo->get_sigma_time()/CLHEP::second);
		    		sigma_isolated_calo_charge.push_back(pCD_calo->get_sigma_charge());
		    		sigma_isolated_calo_amplitude.push_back(pCD_calo->get_sigma_amplitude());
		  		}
			}

			// Add Data from  UDD bank
			const snemo::datamodel::CalorimeterDigiHitHdlCollection & UDD_Calo_hit_Hdl = UDD.get_calorimeter_hits(); 
			for(const auto UDD_CALO : UDD_Calo_hit_Hdl){
				if(std::find(isolated_calo_num.begin(), isolated_calo_num.end(), snemo::datamodel::om_num(UDD_CALO->get_geom_id())) != isolated_calo_num.end()){
					isolated_calo_low_threshold_only.push_back(UDD_CALO->is_low_threshold_only());
					isolated_calo_high_threshold.push_back(UDD_CALO->is_high_threshold());
		    	}
			}

			if(DEBUG == true){
				std::cout<<""<<std::endl;
				std::cout<<"==> DEBUG GAMMA : "<<std::endl;
				std::cout<<"NB ISOLATED CALO = "<<nb_isolated_calo<<std::endl;
				for(int i = 0; i <E_isolated_calo.size(); i++){
			    	std::cout<<"==> TYPE OM = "<<isolated_calo_type.at(i)<<std::endl;
			    	std::cout<<"OM NUM = "<<isolated_calo_num.at(i)<<std::endl;
			    	std::cout<<"E = "<<E_isolated_calo.at(i)<< " +- "<<sigma_E_isolated_calo.at(i)<<" MeV"<<std::endl;
			    	std::cout<<"Time = "<<time_isolated_calo.at(i)<<" +- "<< sigma_time_isolated_calo.at(i)<<" ??"<<std::endl;

			    	std::cout<<"Timestamp = "<<std::setprecision(25)<<isolated_calo_timestamp.at(i)<< " +- "<<std::setprecision(25)<<sigma_isolated_calo_timestamp.at(i)<<std::endl;
			    	std::cout<<"Charge = "<<isolated_calo_charge.at(i)<<" +- "<<sigma_isolated_calo_charge.at(i)<<std::endl;
			    	std::cout<<"Amplitude = "<<isolated_calo_charge.at(i)<< " +- "<<sigma_isolated_calo_amplitude.at(i)<<std::endl;
			    	std::cout <<"is low threshold only : "<<isolated_calo_low_threshold_only.at(i) << std::endl;
			    	std::cout<<"is high threshold : "<< isolated_calo_high_threshold.at(i)<< std::endl;
			  	}	
			}
    	}
    	if(nb_isolated_calo>0){
			evt_isolated_calo=true;
    	}
    	
		// END GAMMA
	
	
    	// BEGINING TRACKS WITH ASSOICATED CALO
    	// Iterate over all particles in the PTD bank
    	//std::cout<<"\n \n"<<std::endl;
    	
    	//std::cout<<"nb particles in the current event : "<<PTD.particles().size()<<std::endl;
		
    	if(PTD.hasParticles()){
			
			
			for(const datatools::handle<snemo::datamodel::particle_track> & particle : PTD.particles()){

		  		if(DEBUG==true){
		    		std::cout<<"\n"<<std::endl;
		    		std::cout<<"proceeding particle ... "<<std::endl;
		    		std::cout<<"track id : "<< particle->get_track_id()<<std::endl;
		    		std::cout<<"trajectory id : "<< particle->get_trajectory().get_id()<<std::endl;
		    		std::cout<<"has alternative tracks "<<PTD.hasAlternativeTracks(particle)<<std::endl;
					std::cout<<"	==> fit infos : Chi2 : " << particle->get_trajectory_handle()->get_fit_infos().get_chi2()<<std::endl;
					std::cout<<"	==> fit infos : t0 : " << particle->get_trajectory_handle()->get_fit_infos().get_t0()<<std::endl;
					std::cout<<"	==> fit infos : Ndof : " << particle->get_trajectory_handle()->get_fit_infos().get_ndof()<<std::endl;
					std::cout<<"	==> fit infos : is best : " << particle->get_trajectory_handle()->get_fit_infos().is_best()<<std::endl;
					std::cout<<"	==> fit infos : get guess : " << particle->get_trajectory_handle()->get_fit_infos().get_guess()<<std::endl;
					std::cout<<"	==> fit infos : get algo : " << particle->get_trajectory_handle()->get_fit_infos().get_algo()<<std::endl;
					std::cout<<"	==> fit infos : first : " << particle->get_trajectory_handle()->get_pattern().get_first()<<std::endl;
					std::cout<<"	==> fit infos : last : " << particle->get_trajectory_handle()->get_pattern().get_last()<<std::endl;
					std::cout<<"	==> has trajectory = "<< particle->has_trajectory()<<std::endl;
					std::cout<<"\n\n";
					particle->print_tree();
					std::cout<<"==================="<<std::endl;
					particle->get_trajectory().print_tree();
					std::cout<<"==================="<<std::endl;
					std::cout<<"ID Cluster of the particle : "<< particle->get_trajectory().get_cluster().get_cluster_id()<<std::endl;
					std::cout<<"\n\n";
					
					for(const datatools::handle<snemo::datamodel::vertex> & vertex : particle->get_vertices()){

						if(vertex->is_on_reference_source_plane()){
							std::cout<<"is_on_reference_source_plane"<<std::endl;
							std::cout<<"	get_distance_xy() = "<<vertex->get_distance_xy()/CLHEP::mm<<std::endl;
							//std::cout<<"	coordinates = "<<vertex->vertex()<<std::endl;
							const geomtools::blur_spot &spot = vertex->get_spot();
							const geomtools::vector_3d &position = spot.get_position(); // https://github.com/BxCppDev/Bayeux/blob/develop/source/bxgeomtools/include/geomtools/blur_spot.h
							std::cout << "Spot coordinates: (" << position[0] << ", " << position.y() << ", " << position.z() << ")" << std::endl;



						}
						if(vertex->is_on_source_foil()){
							std::cout<<"is on source foil"<<std::endl;
							std::cout<<"	get_distance_xy() = "<<vertex->get_distance_xy()/CLHEP::mm<<std::endl;
							std::cout<<"	coordinates = "<<vertex->get_extrapolation()<<std::endl;
						}
						if(vertex->is_on_main_calorimeter()){
							std::cout<<"is_on_main_calorimeter"<<std::endl;
							std::cout<<"	get_distance_xy() = "<<vertex->get_distance_xy()/CLHEP::mm<<std::endl;
							std::cout<<"	coordinates = "<<vertex->get_extrapolation()<<std::endl;
						}
						if(vertex->is_on_x_calorimeter()){
							std::cout<<"is_on_x_calorimeter"<<std::endl;
							std::cout<<"	get_distance_xy() = "<<vertex->get_distance_xy()/CLHEP::mm<<std::endl;
							//std::cout<<"	coordinates = "<<vertex->get_extrapolation()<<" , "<<vertex->get_extrapolation()[1]<<" , "<<vertex->get_extrapolation()[2]<<std::endl;
						}
						if(vertex->is_on_gamma_veto()){
							std::cout<<"is_on_gamma_veto"<<std::endl;
							std::cout<<"	get_distance_xy() = "<<vertex->get_distance_xy()/CLHEP::mm<<std::endl;
							std::cout<<"	coordinates = "<<vertex->get_extrapolation()<<std::endl;
						}
					}
					
					// Here trying to extrac info on alternative tracks for a given particle 
					//const auto TTD_solution = TTD.solution();
					std::cout<<"TTD.get_number_of_solutions() = "<< TTD.get_number_of_solutions()<<std::endl;
					const auto TTD_solution = TTD.get_solutions();

					for(const auto a : TTD_solution){
						std::cout<<"==> TTD PRINT"<< "\n"<<std::endl;
						a->print_tree();
					}	
			  	}

					
				

		  		//Check if the particle has an associated calorimet hit (or more) ==> electron candidates ! 
		  		if(particle->has_associated_calorimeter_hits() ){ 
					
		    		vector<double> cell_num_of_current_track;
		    		vector<double> OM_num_of_current_track;
		    		vector<double> E_OM_of_current_track;
		    		vector<double> timestamp_OM_of_current_track;
		    		vector<double> OM_low_threshold_only_of_current_track;
		    		vector<double> OM_high_treshold_of_current_track;
		    		int nb_OM=0;

		    		const auto track_cluster = particle->get_trajectory_handle()->get_cluster(); // PTD -> particle_track_data.h -> particle_track.h (get_trajectory_handle()) -> tracker_trajectory.h (get_cluster()) -> tracker_cluster.h 
		    		const auto track_hits = track_cluster.hits();
		    		const auto pCD_tracker_hits = PCD.tracker_hits();
					
					vector<double> z_of_cell_for_current_track;
					vector<double> sigma_z_of_cell_for_current_track;
		    		vector<double>r_of_current_track, sigma_r_of_current_track;
					
					
					elec_cluster_is_delayed.push_back(track_cluster.is_delayed());
					elec_clsuter_is_prompt.push_back(track_cluster.is_prompt());
					//std::cout<<" cluster ID = "<<track_cluster.get_cluster_id()<<std::endl;
					ID_clsuter_per_elec_cluster.push_back(track_cluster.get_cluster_id());
					

					// PTD Bank for tracker hits
					for(const auto hits : track_hits){// looping on all GG hit of the cluster
					 	double PTD_id = hits->get_id();

		      			cell_num_of_current_track.push_back(snemo::datamodel::gg_num(hits->get_geom_id()));
						 
						z_of_cell_for_current_track.push_back(hits->get_z()/CLHEP::mm); // set in mm 
						sigma_z_of_cell_for_current_track.push_back(hits->get_sigma_z()/CLHEP::mm); // set in mm
						
						r_of_current_track.push_back(hits->get_r());
						sigma_r_of_current_track.push_back(hits->get_sigma_r()); 
					
					}

		    		cell_num_per_elec_cluster.push_back(cell_num_of_current_track);
	    			
					
					
	    			vector<double> R0_for_current_track;
	    			vector<double> Top_cathode_for_current_track;
	    			vector<double> Bottom_cathode_for_current_track;
					
					// pCD Bank tracker hits
	    			for(const auto pCD_track_hits : pCD_tracker_hits){
						if(std::find(cell_num_of_current_track.begin(), cell_num_of_current_track.end(), snemo::datamodel::gg_num(pCD_track_hits->get_geom_id())) != cell_num_of_current_track.end() ){

							R0_for_current_track.push_back(pCD_track_hits->get_anodic_time()/ CLHEP::second); // en us ?? WTF je vois ça comme si c'était en s 
							Top_cathode_for_current_track.push_back(pCD_track_hits->get_top_cathode_drift_time()/ CLHEP::second);
							Bottom_cathode_for_current_track.push_back(pCD_track_hits->get_bottom_cathode_drift_time()/ CLHEP::second);



	      				}
	      
	    			}


					z_of_cells_per_elec_cluster.push_back(z_of_cell_for_current_track);
					sigma_z_of_cells_per_elec_cluster.push_back(sigma_z_of_cell_for_current_track);
					anode_time_per_elec_cluster.push_back(R0_for_current_track);
	      			top_cathode_per_elec_cluster.push_back(Top_cathode_for_current_track);
	      			bottom_cathode_per_elec_cluster.push_back(Bottom_cathode_for_current_track);
	      			R0_for_current_track.clear();
	      			Top_cathode_for_current_track.clear();
	      			Bottom_cathode_for_current_track.clear();
	    			cell_num_of_current_track.clear();
					z_of_cell_for_current_track.clear();
					sigma_z_of_cell_for_current_track.clear();
					r_of_cells_per_elec_cluster.push_back(r_of_current_track);
					sigma_r_of_cells_per_elec_cluster.push_back(sigma_r_of_current_track);
					r_of_current_track.clear();
					sigma_r_of_current_track.clear();


	    			// Recovering the OM data from PTD bank
	    
	    			const auto & PTD_calo_hits = particle->get_associated_calorimeter_hits();
	    			for(const auto & calo_hit : PTD_calo_hits){
	      				nb_OM++;
	      				OM_num_of_current_track.push_back(snemo::datamodel::om_num(calo_hit->get_geom_id()));
	      				E_OM_of_current_track.push_back(calo_hit->get_energy());
	    			}
					OM_num_per_elec_cluster.push_back(OM_num_of_current_track);
	    			E_OM_per_elec_cluster.push_back(E_OM_of_current_track);
	    			nb_of_OM_per_elec_cluster.push_back(nb_OM);
					
					// NEED TO EXTRACT pCD informations of calo ! 
					
					vector<double> OM_timestamp_of_current_track, OM_charge_of_current_track, OM_amplitude_of_current_track;

					const auto pCD_calo_with_tracks = PCD.calorimeter_hits();
					for(const auto & pCD_calo_hit : pCD_calo_with_tracks){
						if(std::find(OM_num_of_current_track.begin(), OM_num_of_current_track.end(), snemo::datamodel::om_num(pCD_calo_hit->get_geom_id())) != OM_num_of_current_track.end()){
							OM_timestamp_of_current_track.push_back(pCD_calo_hit->get_time()/CLHEP::second);
							OM_charge_of_current_track.push_back(pCD_calo_hit->get_charge());
							OM_amplitude_of_current_track.push_back(pCD_calo_hit->get_amplitude());

						}
					}

					OM_timestamp_per_elec_cluster.push_back(OM_timestamp_of_current_track); // in us
					OM_charge_per_elec_cluster.push_back(OM_charge_of_current_track);
					OM_amplitude_per_elec_cluster.push_back(OM_amplitude_of_current_track);
					OM_timestamp_of_current_track.clear();
					OM_charge_of_current_track.clear();
					OM_amplitude_of_current_track.clear();
					
	    			OM_num_of_current_track.clear();
	    			E_OM_of_current_track.clear();
	    			nb_OM=0;
					
					if(particle->has_trajectory()){
						x_start_per_elec_cluster.push_back({particle->get_trajectory_handle()->get_pattern().get_first()[0]});
						y_start_per_elec_cluster.push_back({particle->get_trajectory_handle()->get_pattern().get_first()[1]});
						z_start_per_elec_cluster.push_back({particle->get_trajectory_handle()->get_pattern().get_first()[2]});
						x_end_per_elec_cluster.push_back({particle->get_trajectory_handle()->get_pattern().get_last()[0]});
						y_end_per_elec_cluster.push_back({particle->get_trajectory_handle()->get_pattern().get_last()[1]});
						z_end_per_elec_cluster.push_back({particle->get_trajectory_handle()->get_pattern().get_last()[2]});
					}
					else{
						x_start_per_elec_cluster.push_back({std::nan("")});
						y_start_per_elec_cluster.push_back({std::nan("")});
						z_start_per_elec_cluster.push_back({std::nan("")});
						x_end_per_elec_cluster.push_back({std::nan("")});
						y_end_per_elec_cluster.push_back({std::nan("")});
						z_end_per_elec_cluster.push_back({std::nan("")});
					}
					
					// Lopping on all extrapolate vertexes and save it ! 


					//std::cout<<"ICI ? "<<std::endl;
					if(particle->has_trajectory() && particle->has_vertices()){
						// Temporary vectors to store vertex data for the current particle
    					vector<double> temp_x_is_on_reference_source_plane(1, std::nan(""));
    					vector<double> temp_y_is_on_reference_source_plane(1, std::nan(""));
    					vector<double> temp_z_is_on_reference_source_plane(1, std::nan(""));
    					vector<double> temp_x_is_on_source_foil(1, std::nan(""));
    					vector<double> temp_y_is_on_source_foil(1, std::nan(""));
    					vector<double> temp_z_is_on_source_foil(1, std::nan(""));
    					vector<double> temp_x_is_on_main_calorimeter(1, std::nan(""));
    					vector<double> temp_y_is_on_main_calorimeter(1, std::nan(""));
    					vector<double> temp_z_is_on_main_calorimeter(1, std::nan(""));
    					vector<double> temp_x_is_on_x_calorimeter(1, std::nan(""));
    					vector<double> temp_y_is_on_x_calorimeter(1, std::nan(""));
    					vector<double> temp_z_is_on_x_calorimeter(1, std::nan(""));
    					vector<double> temp_x_is_on_gamma_veto(1, std::nan(""));
    					vector<double> temp_y_is_on_gamma_veto(1, std::nan(""));
    					vector<double> temp_z_is_on_gamma_veto(1, std::nan(""));
    					vector<double> temp_x_is_on_wire(1, std::nan(""));
    					vector<double> temp_y_is_on_wire(1, std::nan(""));
    					vector<double> temp_z_is_on_wire(1, std::nan(""));
    					vector<double> temp_x_is_in_gas(1, std::nan(""));
    					vector<double> temp_y_is_in_gas(1, std::nan(""));
    					vector<double> temp_z_is_in_gas(1, std::nan(""));
    					vector<double> temp_x_is_on_source_gap(1, std::nan(""));
    					vector<double> temp_y_is_on_source_gap(1, std::nan(""));
    					vector<double> temp_z_is_on_source_gap(1, std::nan(""));
    					vector<double> temp_xy_distance_from_reference_source_plane(1, std::nan(""));
    					vector<double> temp_xy_distance_from_source_foil(1, std::nan(""));
    					vector<double> temp_xy_distance_from_main_calorimeter(1, std::nan(""));
    					vector<double> temp_xy_distance_from_x_calorimeter(1, std::nan(""));
    					vector<double> temp_xy_distance_from_gamma_veto(1, std::nan(""));
    					vector<double> temp_xy_distance_from_wire(1, std::nan(""));
    					vector<double> temp_xy_distance_from_gas(1, std::nan(""));
    					vector<double> temp_xy_distance_from_source_gap(1, std::nan(""));
    					vector<double> temp_xyz_distance_from_reference_source_plane(1, std::nan(""));
    					vector<double> temp_xyz_distance_from_source_foil(1, std::nan(""));
    					vector<double> temp_xyz_distance_from_main_calorimeter(1, std::nan(""));
    					vector<double> temp_xyz_distance_from_x_calorimeter(1, std::nan(""));
    					vector<double> temp_xyz_distance_from_gamma_veto(1, std::nan(""));
    					vector<double> temp_xyz_distance_from_wire(1, std::nan(""));
    					vector<double> temp_xyz_distance_from_gas(1, std::nan(""));
    					vector<double> temp_xyz_distance_from_source_gap(1, std::nan(""));

						vector<double> temp_vertex_is_on_reference_source_plane_per_elec_cluster(1, 0);
						vector<double> temp_vertex_is_on_source_foil_per_elec_cluster(1, 0);
						vector<double> temp_vertex_is_on_main_calorimeter_per_elec_cluster(1, 0);
						vector<double> temp_vertex_is_on_x_calorimeter_per_elec_cluster(1, 0);
						vector<double> temp_vertex_is_on_gamma_veto_per_elec_cluster(1, 0);
						vector<double> temp_vertex_is_on_wire_per_elec_cluster(1, 0);
						vector<double> temp_vertex_is_in_gas_per_elec_cluster(1, 0);
						vector<double> temp_vertex_is_on_source_gap_per_elec_cluster(1, 0);

						for(const datatools::handle<snemo::datamodel::vertex> & vertex : particle->get_vertices()){
							const geomtools::blur_spot &spot = vertex->get_spot();
							const geomtools::vector_3d &position = spot.get_position(); // https://github.com/BxCppDev/Bayeux/blob/develop/source/bxgeomtools/include/geomtools/blur_spot.h
							
							if(DEBUG == true){
								std::cout << "vertex->is_on_reference_source_plane: " << vertex->is_on_reference_source_plane() << std::endl;
    							std::cout << "vertex->is_on_source_foil: " << vertex->is_on_source_foil() << std::endl;
    							std::cout << "vertex->is_on_main_calorimeter: " << vertex->is_on_main_calorimeter() << std::endl;
    							std::cout << "vertex->is_on_x_calorimeter: " << vertex->is_on_x_calorimeter() << std::endl;
    							std::cout << "vertex->is_on_gamma_veto: " << vertex->is_on_gamma_veto() << std::endl;
    							std::cout << "vertex->is_on_wire: " << vertex->is_on_wire() << std::endl;
    							std::cout << "vertex->is_in_gas: " << vertex->is_in_gas() << std::endl;
    							std::cout << "vertex->is_on_source_gap: " << vertex->is_on_source_gap() << std::endl;
								//bool elec_is_on_reference_source_plane = false;
								// elec_is_on_source_foil
								// elec_is_on_main_calorimeter
								// elec_is_on_x_calorimeter
								// elec_is_on_gamma_veto
								// elec_is_on_wire
								// elec_is_in_gas
								// elec_is_on_source_gap

								
								//temp_vertex_is_on_reference_source_plane_per_elec_cluster
								//temp_vertex_is_on_source_foil_per_elec_cluster
								//temp_vertex_is_on_main_calorimeter_per_elec_cluster
								//temp_vertex_is_on_x_calorimeter_per_elec_cluster
								//temp_vertex_is_on_gamma_veto_per_elec_cluster
								//temp_vertex_is_on_wire_per_elec_cluster
								//temp_vertex_is_in_gas_per_elec_cluster
								//temp_vertex_is_on_source_gap_per_elec_cluster
							}

							if(vertex->is_on_reference_source_plane()){
								elec_is_on_reference_source_plane = true;
								temp_vertex_is_on_reference_source_plane_per_elec_cluster[0] = 1;
								temp_x_is_on_reference_source_plane[0] = position[0];
								temp_y_is_on_reference_source_plane[0] = position[1];
								temp_z_is_on_reference_source_plane[0] = position[2];
								temp_xy_distance_from_reference_source_plane[0] = vertex->get_distance_xy()/CLHEP::mm;
								temp_xyz_distance_from_reference_source_plane[0] = vertex->get_distance()/CLHEP::mm;

							}
							
							
							if(vertex->is_on_source_foil()){
								elec_is_on_source_foil = true;
								temp_vertex_is_on_source_foil_per_elec_cluster[0] = (1);
								temp_x_is_on_source_foil[0] = position[0];
								temp_y_is_on_source_foil[0] = position[1];
								temp_z_is_on_source_foil[0] = position[2];
								temp_xy_distance_from_source_foil[0] = vertex->get_distance_xy()/CLHEP::mm;
								temp_xyz_distance_from_source_foil[0] = vertex->get_distance()/CLHEP::mm;
							}
							
							if(vertex->is_on_main_calorimeter()){
								elec_is_on_main_calorimeter = true;
								temp_vertex_is_on_main_calorimeter_per_elec_cluster[0] = 1;
								temp_x_is_on_main_calorimeter[0] = position[0];
								temp_y_is_on_main_calorimeter[0] = position[1];
								temp_z_is_on_main_calorimeter[0] = position[2];
								temp_xy_distance_from_main_calorimeter[0] = vertex->get_distance_xy()/CLHEP::mm;
								temp_xyz_distance_from_main_calorimeter[0] = vertex->get_distance()/CLHEP::mm;
							}
							
							if(vertex->is_on_x_calorimeter()){
								elec_is_on_x_calorimeter = true;
								temp_vertex_is_on_x_calorimeter_per_elec_cluster[0] = 1;
								temp_x_is_on_x_calorimeter[0] = position[0];
								temp_y_is_on_x_calorimeter[0] = position[1];
								temp_z_is_on_x_calorimeter[0] = position[2];
								temp_xy_distance_from_x_calorimeter[0]  = vertex->get_distance_xy()/CLHEP::mm;
								temp_xyz_distance_from_x_calorimeter[0] = vertex->get_distance()/CLHEP::mm;
							}
							
							if(vertex->is_on_gamma_veto()){
								elec_is_on_gamma_veto = true;
								temp_vertex_is_on_gamma_veto_per_elec_cluster[0] = 1;
								temp_x_is_on_gamma_veto[0] = position[0];
								temp_y_is_on_gamma_veto[0] = position[1];
								temp_z_is_on_gamma_veto[0] = position[2];
								temp_xy_distance_from_gamma_veto[0] = vertex->get_distance_xy()/CLHEP::mm;
								temp_xyz_distance_from_gamma_veto[0] = vertex->get_distance()/CLHEP::mm;
							}	
							
							if(vertex->is_on_wire()){
								elec_is_on_wire = true;
								temp_vertex_is_on_wire_per_elec_cluster[0] = 1;
								temp_x_is_on_wire[0] = position[0];
								temp_y_is_on_wire[0] = position[1];
								temp_z_is_on_wire[0] = position[2];
								temp_xy_distance_from_wire[0] = vertex->get_distance_xy()/CLHEP::mm;
								temp_xyz_distance_from_wire[0] = vertex->get_distance()/CLHEP::mm;
							}
							
							if(vertex->is_in_gas()){
								elec_is_in_gas = true;
								temp_vertex_is_in_gas_per_elec_cluster[0]= 1;
								temp_x_is_in_gas[0]= position[0];
								temp_y_is_in_gas[0]= position[1];
								temp_z_is_in_gas[0]= position[2];
								temp_xy_distance_from_gas[0]= vertex->get_distance_xy()/CLHEP::mm;
								temp_xyz_distance_from_gas[0]= vertex->get_distance()/CLHEP::mm;
							}
							
							if(vertex->is_on_source_gap()){
								elec_is_on_source_gap = true;
								temp_vertex_is_on_source_gap_per_elec_cluster[0] = 1;
								temp_x_is_on_source_gap[0] = position[0];
								temp_y_is_on_source_gap[0] = position[1];
								temp_z_is_on_source_gap[0] = position[2];
								temp_xy_distance_from_source_gap[0] =  vertex->get_distance_xy()/CLHEP::mm;
								temp_xyz_distance_from_source_gap[0] = vertex->get_distance()/CLHEP::mm;
							}
							
						}
						// Push the collected data for this particle into the main vectors
    					x_is_on_reference_source_plane.push_back(temp_x_is_on_reference_source_plane);
    					y_is_on_reference_source_plane.push_back(temp_y_is_on_reference_source_plane);
    					z_is_on_reference_source_plane.push_back(temp_z_is_on_reference_source_plane);
    					x_is_on_source_foil.push_back(temp_x_is_on_source_foil);
    					y_is_on_source_foil.push_back(temp_y_is_on_source_foil);
    					z_is_on_source_foil.push_back(temp_z_is_on_source_foil);
    					x_is_on_main_calorimeter.push_back(temp_x_is_on_main_calorimeter);
    					y_is_on_main_calorimeter.push_back(temp_y_is_on_main_calorimeter);
    					z_is_on_main_calorimeter.push_back(temp_z_is_on_main_calorimeter);
    					x_is_on_x_calorimeter.push_back(temp_x_is_on_x_calorimeter);
    					y_is_on_x_calorimeter.push_back(temp_y_is_on_x_calorimeter);
    					z_is_on_x_calorimeter.push_back(temp_z_is_on_x_calorimeter);
    					x_is_on_gamma_veto.push_back(temp_x_is_on_gamma_veto);
    					y_is_on_gamma_veto.push_back(temp_y_is_on_gamma_veto);
    					z_is_on_gamma_veto.push_back(temp_z_is_on_gamma_veto);
    					x_is_on_wire.push_back(temp_x_is_on_wire);
    					y_is_on_wire.push_back(temp_y_is_on_wire);
    					z_is_on_wire.push_back(temp_z_is_on_wire);
    					x_is_in_gas.push_back(temp_x_is_in_gas);
    					y_is_in_gas.push_back(temp_y_is_in_gas);
    					z_is_in_gas.push_back(temp_z_is_in_gas);
    					x_is_on_source_gap.push_back(temp_x_is_on_source_gap);
    					y_is_on_source_gap.push_back(temp_y_is_on_source_gap);
    					z_is_on_source_gap.push_back(temp_z_is_on_source_gap);
    					xy_distance_from_reference_source_plane.push_back(temp_xy_distance_from_reference_source_plane);
    					xy_distance_from_source_foil.push_back(temp_xy_distance_from_source_foil);
    					xy_distance_from_main_calorimeter.push_back(temp_xy_distance_from_main_calorimeter);
    					xy_distance_from_x_calorimeter.push_back(temp_xy_distance_from_x_calorimeter);
    					xy_distance_from_gamma_veto.push_back(temp_xy_distance_from_gamma_veto);
    					xy_distance_from_wire.push_back(temp_xy_distance_from_wire);
    					xy_distance_from_gas.push_back(temp_xy_distance_from_gas);
    					xy_distance_from_source_gap.push_back(temp_xy_distance_from_source_gap);
    					xyz_distance_from_reference_source_plane.push_back(temp_xyz_distance_from_reference_source_plane);
    					xyz_distance_from_source_foil.push_back(temp_xyz_distance_from_source_foil);
    					xyz_distance_from_main_calorimeter.push_back(temp_xyz_distance_from_main_calorimeter);
    					xyz_distance_from_x_calorimeter.push_back(temp_xyz_distance_from_x_calorimeter);
    					xyz_distance_from_gamma_veto.push_back(temp_xyz_distance_from_gamma_veto);
    					xyz_distance_from_wire.push_back(temp_xyz_distance_from_wire);
    					xyz_distance_from_gas.push_back(temp_xyz_distance_from_gas);
    					xyz_distance_from_source_gap.push_back(temp_xyz_distance_from_source_gap);
						vertex_is_on_reference_source_plane_per_elec_cluster.push_back(temp_vertex_is_on_reference_source_plane_per_elec_cluster);
						vertex_is_on_source_foil_per_elec_cluster.push_back(temp_vertex_is_on_source_foil_per_elec_cluster);
						vertex_is_on_main_calorimeter_per_elec_cluster.push_back(temp_vertex_is_on_main_calorimeter_per_elec_cluster);
						vertex_is_on_x_calorimeter_per_elec_cluster.push_back(temp_vertex_is_on_x_calorimeter_per_elec_cluster);
						vertex_is_on_gamma_veto_per_elec_cluster.push_back(temp_vertex_is_on_gamma_veto_per_elec_cluster);
						vertex_is_on_wire_per_elec_cluster.push_back(temp_vertex_is_on_wire_per_elec_cluster);
						vertex_is_in_gas_per_elec_cluster.push_back(temp_vertex_is_in_gas_per_elec_cluster);
						vertex_is_on_source_gap_per_elec_cluster.push_back(temp_vertex_is_on_source_gap_per_elec_cluster);


					}

					else{
						 
						x_is_on_source_foil.push_back({std::nan("")});
						y_is_on_source_foil.push_back({std::nan("")});
						z_is_on_source_foil.push_back({std::nan("")});
						xy_distance_from_source_foil.push_back({std::nan("")});
						xyz_distance_from_source_foil.push_back({std::nan("")});

						 
						x_is_on_source_foil.push_back({std::nan("")});
						y_is_on_source_foil.push_back({std::nan("")});
						z_is_on_source_foil.push_back({std::nan("")});
						xy_distance_from_source_foil.push_back({std::nan("")});
						xyz_distance_from_source_foil.push_back({std::nan("")});

						 
						x_is_on_main_calorimeter.push_back({std::nan("")});
						y_is_on_main_calorimeter.push_back({std::nan("")});
						z_is_on_main_calorimeter.push_back({std::nan("")});
						xy_distance_from_main_calorimeter.push_back({std::nan("")});
						xyz_distance_from_main_calorimeter.push_back({std::nan("")});


						 
						x_is_on_x_calorimeter.push_back({std::nan("")});
						y_is_on_x_calorimeter.push_back({std::nan("")});
						z_is_on_x_calorimeter.push_back({std::nan("")});
						xy_distance_from_x_calorimeter.push_back({std::nan("")});
						xyz_distance_from_x_calorimeter.push_back({std::nan("")});


						 
						x_is_on_gamma_veto.push_back({std::nan("")});
						y_is_on_gamma_veto.push_back({std::nan("")});
						z_is_on_gamma_veto.push_back({std::nan("")});
						xy_distance_from_gamma_veto.push_back({std::nan("")});
						xyz_distance_from_gamma_veto.push_back({std::nan("")});

						 
						x_is_on_wire.push_back({std::nan("")});
						y_is_on_wire.push_back({std::nan("")});
						z_is_on_wire.push_back({std::nan("")});
						xy_distance_from_wire.push_back({std::nan("")});
						xyz_distance_from_wire.push_back({std::nan("")});
						
						
						 
						x_is_in_gas.push_back({std::nan("")});
						y_is_in_gas.push_back({std::nan("")});
						z_is_in_gas.push_back({std::nan("")});
						xy_distance_from_gas.push_back({std::nan("")});
						xyz_distance_from_gas.push_back({std::nan("")});

						 
						x_is_on_source_gap.push_back({std::nan("")});
						y_is_on_source_gap.push_back({std::nan("")});
						z_is_on_source_gap.push_back({std::nan("")});
						xy_distance_from_source_gap.push_back({std::nan("")});
						xyz_distance_from_source_gap.push_back({std::nan("")});


						vertex_is_on_reference_source_plane_per_elec_cluster.push_back({0});
						vertex_is_on_source_foil_per_elec_cluster.push_back({0});
						vertex_is_on_main_calorimeter_per_elec_cluster.push_back({0});
						vertex_is_on_x_calorimeter_per_elec_cluster.push_back({0});
						vertex_is_on_gamma_veto_per_elec_cluster.push_back({0});
						vertex_is_on_wire_per_elec_cluster.push_back({0});
						vertex_is_in_gas_per_elec_cluster.push_back({0});
						vertex_is_on_source_gap_per_elec_cluster.push_back({0});
					}
	  			}
				
			}


			

			//std::cout<<"END ELEC "<<std::endl;
			//END PARTICLE LOOP

			// RECOVERING OTHER INFORMATION FOR ELECTRON LIKE 
			
			// Time to delete some vectors in the case where there are 2 particle associated to the same cluster. for exemple, we can check directly cluster ID are the same or are already analyzed in the previous loop if particle, or 
			// at the END ckeck if few vectors in cell_num_per_elec_cluster are equal to each other. If it is the case, delete the second cluster equal to the first, and delete all other information associated with this cluster.

			// May be we can do it really easy by checking the cluster ID if it already exist... 
			

			
			// Here to avoide tu put again the same informations, because we are not in the particle loop
			const snemo::datamodel::CalorimeterDigiHitHdlCollection & UDD_Calo_hit_Hdl = UDD.get_calorimeter_hits(); 
			for(int h = 0; h < OM_num_per_elec_cluster.size(); h++){
				vector<double> LT_only_of_current_cluster, HT_of_current_cluster;
				for(const auto UDD_CALO : UDD_Calo_hit_Hdl){
					if(std::find(OM_num_per_elec_cluster[h].begin(), OM_num_per_elec_cluster[h].end(), snemo::datamodel::om_num(UDD_CALO->get_geom_id())) != OM_num_per_elec_cluster[h].end()){
						if(UDD_CALO->is_low_threshold_only()){
							LT_only_of_current_cluster.push_back(1);
						}
						else{
							LT_only_of_current_cluster.push_back(0);
						}
						if(UDD_CALO->is_high_threshold()){
							HT_of_current_cluster.push_back(1);
						}
						else{
							HT_of_current_cluster.push_back(0);
						}
						
		    		}
				}
				OM_LT_only_per_elec_cluster.push_back(LT_only_of_current_cluster);
				OM_HT_per_elec_cluster.push_back(HT_of_current_cluster);
				LT_only_of_current_cluster.clear();
				HT_of_current_cluster.clear();
			}
			
			// Check if 2 particles hase the same cluster, if yes, erase one
			
			

			for (size_t i = 0; i < cell_num_per_elec_cluster.size(); ++i) {
        		for (size_t j = i + 1; j < cell_num_per_elec_cluster.size(); ) {
            		
					if(DEBUG == true){
						std::cout<<"LA ! "<<std::endl;
						std::cout<<"x_is_on_source_foil[i][0] = "<<x_is_on_source_foil[i][0]<<std::endl;
						std::cout<<"x_is_on_reference_source_plane[i][0] = "<< x_is_on_reference_source_plane[i][0] <<std::endl;
						std::cout<<"x_is_on_main_calorimeter[i][0] = "<< x_is_on_main_calorimeter[i][0] <<std::endl;
						std::cout<<"x_is_on_x_calorimeter[i][0] = "<< x_is_on_x_calorimeter[i][0]<<std::endl;
						std::cout<<"x_is_on_gamma_veto[i][0] = "<< x_is_on_gamma_veto[i][0]<<std::endl;
						std::cout<<"x_is_on_wire[i][0] = "<<x_is_on_wire[i][0] <<std::endl;
						std::cout<<"x_is_in_gas[i][0] = "<< x_is_in_gas[i][0]<<std::endl;
						std::cout<<"x_is_on_source_gap[i][0] = "<< x_is_on_source_gap[i][0]<<std::endl;

					}
					
					
					if (cell_num_per_elec_cluster[i]== cell_num_per_elec_cluster[j]) {
                		
                		// Remove duplicate
                		cell_num_per_elec_cluster.erase(cell_num_per_elec_cluster.begin() + j);
						OM_num_per_elec_cluster.erase(OM_num_per_elec_cluster.begin() + j);
  						E_OM_per_elec_cluster.erase(E_OM_per_elec_cluster.begin() + j);
  						nb_of_OM_per_elec_cluster.erase(nb_of_OM_per_elec_cluster.begin() + j);
  						anode_time_per_elec_cluster.erase(anode_time_per_elec_cluster.begin() + j);
  						top_cathode_per_elec_cluster.erase(top_cathode_per_elec_cluster.begin() + j);
  						bottom_cathode_per_elec_cluster.erase(bottom_cathode_per_elec_cluster.begin() + j);
						OM_timestamp_per_elec_cluster.erase(OM_timestamp_per_elec_cluster.begin() + j);
						OM_charge_per_elec_cluster.erase(OM_charge_per_elec_cluster.begin() + j);
						OM_amplitude_per_elec_cluster.erase(OM_amplitude_per_elec_cluster.begin() + j);
						OM_LT_only_per_elec_cluster.erase(OM_LT_only_per_elec_cluster.begin() + j);
						OM_HT_per_elec_cluster.erase(OM_HT_per_elec_cluster.begin() + j);
						z_of_cells_per_elec_cluster.erase(z_of_cells_per_elec_cluster.begin() + j);
						sigma_z_of_cells_per_elec_cluster.erase(sigma_z_of_cells_per_elec_cluster.begin() + j);
						r_of_cells_per_elec_cluster.erase(r_of_cells_per_elec_cluster.begin() + j);
						sigma_r_of_cells_per_elec_cluster.erase(sigma_r_of_cells_per_elec_cluster.begin() + j);
						elec_cluster_is_delayed.erase(elec_cluster_is_delayed.begin() + j);
						elec_clsuter_is_prompt.erase(elec_clsuter_is_prompt.begin() + j);
						
						//std::cout<<"x_start_per_elec_cluster.size()"<<x_start_per_elec_cluster.size()<<std::endl;
						//std::cout<<"x_start_per_elec_cluster[i].size()"<<x_start_per_elec_cluster[i].size()<<std::endl;
						x_start_per_elec_cluster[i].push_back(x_start_per_elec_cluster[j][0]);
						y_start_per_elec_cluster[i].push_back(y_start_per_elec_cluster[j][0]);
						z_start_per_elec_cluster[i].push_back(z_start_per_elec_cluster[j][0]);
						x_end_per_elec_cluster[i].push_back(x_end_per_elec_cluster[j][0]);
						y_end_per_elec_cluster[i].push_back(y_end_per_elec_cluster[j][0]);
						z_end_per_elec_cluster[i].push_back(z_end_per_elec_cluster[j][0]);
						//std::cout<<"x_start_per_elec_cluster.size()"<<x_start_per_elec_cluster.size()<<std::endl;
						//std::cout<<"x_start_per_elec_cluster[i].size()"<<x_start_per_elec_cluster[i].size()<<std::endl;
						
						//std::cout<<"LA ALORS ? "<<std::endl;
						
						//std::cout<<"x_is_on_reference_source_plane.size() = "<< x_is_on_reference_source_plane.size()<<std::endl;
						//std::cout<<"x_is_on_reference_source_plane[i].size() = "<< x_is_on_reference_source_plane[i].size()<<std::endl;
						//std::cout<<"x_is_on_reference_source_plane[i][0] = "<< x_is_on_reference_source_plane[i][0]<<std::endl;
						//std::cout<<"y_is_on_reference_source_plane[i][0] = "<< y_is_on_reference_source_plane[i][0]<<std::endl;
						//std::cout<<"z_is_on_reference_source_plane[i][0] = "<< z_is_on_reference_source_plane[i][0]<<std::endl;
						x_is_on_reference_source_plane[i].push_back(x_is_on_reference_source_plane[j][0]);
						y_is_on_reference_source_plane[i].push_back(y_is_on_reference_source_plane[j][0]);
						z_is_on_reference_source_plane[i].push_back(z_is_on_reference_source_plane[j][0]);
						
						//std::cout<<"x_is_on_source_foil[i][0] = "<< x_is_on_source_foil[i][0]<<std::endl;
						//std::cout<<"y_is_on_source_foil[i][0] = "<< y_is_on_source_foil[i][0]<<std::endl;
						//std::cout<<"z_is_on_source_foil[i][0] = "<< z_is_on_source_foil[i][0]<<std::endl;


						x_is_on_source_foil[i].push_back(x_is_on_source_foil[j][0]);
						//std::cout<<"et la ? "<<std::endl;
						y_is_on_source_foil[i].push_back(y_is_on_source_foil[j][0]);
						z_is_on_source_foil[i].push_back(z_is_on_source_foil[j][0]);
						x_is_on_main_calorimeter[i].push_back(x_is_on_main_calorimeter[j][0]);
						y_is_on_main_calorimeter[i].push_back(y_is_on_main_calorimeter[j][0]);
						z_is_on_main_calorimeter[i].push_back(z_is_on_main_calorimeter[j][0]);
						x_is_on_x_calorimeter[i].push_back(x_is_on_x_calorimeter[j][0]);
						
						y_is_on_x_calorimeter[i].push_back(y_is_on_x_calorimeter[j][0]);
						z_is_on_x_calorimeter[i].push_back(z_is_on_x_calorimeter[j][0]);
						x_is_on_gamma_veto[i].push_back(x_is_on_gamma_veto[j][0]);
						y_is_on_gamma_veto[i].push_back(y_is_on_gamma_veto[j][0]);
						z_is_on_gamma_veto[i].push_back(z_is_on_gamma_veto[j][0]);
						
						x_is_on_wire[i].push_back(x_is_on_wire[j][0]);
						y_is_on_wire[i].push_back(y_is_on_wire[j][0]);
						z_is_on_wire[i].push_back(z_is_on_wire[j][0]);
						//std::cout<<"	aled !!"<<std::endl;
						x_is_in_gas[i].push_back(x_is_in_gas[j][0]);
						y_is_in_gas[i].push_back(y_is_in_gas[j][0]);
						z_is_in_gas[i].push_back(z_is_in_gas[j][0]);
						x_is_on_source_gap[i].push_back(x_is_on_source_gap[j][0]);
						y_is_on_source_gap[i].push_back(y_is_on_source_gap[j][0]);
						z_is_on_source_gap[i].push_back(z_is_on_source_gap[j][0]);

						//std::cout<<"pas la ... "<<std::endl;


						xy_distance_from_reference_source_plane[i].push_back(xy_distance_from_reference_source_plane[j][0]);
						xy_distance_from_source_foil[i].push_back(xy_distance_from_source_foil[j][0]);
						xy_distance_from_main_calorimeter[i].push_back(xy_distance_from_main_calorimeter[j][0]);
						xy_distance_from_x_calorimeter[i].push_back(xy_distance_from_x_calorimeter[j][0]);
						xy_distance_from_gamma_veto[i].push_back(xy_distance_from_gamma_veto[j][0]);
						xy_distance_from_wire[i].push_back(xy_distance_from_wire[j][0]);
						xy_distance_from_gas[i].push_back(xy_distance_from_gas[j][0]);
						xy_distance_from_source_gap[i].push_back(xy_distance_from_source_gap[j][0]);

						xyz_distance_from_reference_source_plane[i].push_back(xyz_distance_from_reference_source_plane[j][0]);
						xyz_distance_from_source_foil[i].push_back(xyz_distance_from_source_foil[j][0]);
						xyz_distance_from_main_calorimeter[i].push_back(xyz_distance_from_main_calorimeter[j][0]);
						xyz_distance_from_x_calorimeter[i].push_back(xyz_distance_from_x_calorimeter[j][0]);
						xyz_distance_from_gamma_veto[i].push_back(xyz_distance_from_gamma_veto[j][0]);
						xyz_distance_from_wire[i].push_back(xyz_distance_from_wire[j][0]);
						xyz_distance_from_gas[i].push_back(xyz_distance_from_gas[j][0]);
						xyz_distance_from_source_gap[i].push_back(xyz_distance_from_source_gap[j][0]);


						vertex_is_on_reference_source_plane_per_elec_cluster[i].push_back(vertex_is_on_reference_source_plane_per_elec_cluster[j][0]);
						vertex_is_on_source_foil_per_elec_cluster[i].push_back(vertex_is_on_source_foil_per_elec_cluster[j][0]);
						vertex_is_on_main_calorimeter_per_elec_cluster[i].push_back(vertex_is_on_main_calorimeter_per_elec_cluster[j][0]);
						vertex_is_on_x_calorimeter_per_elec_cluster[i].push_back(vertex_is_on_x_calorimeter_per_elec_cluster[j][0]);
						vertex_is_on_gamma_veto_per_elec_cluster[i].push_back(vertex_is_on_gamma_veto_per_elec_cluster[j][0]);
						vertex_is_on_wire_per_elec_cluster[i].push_back(vertex_is_on_wire_per_elec_cluster[j][0]);
						vertex_is_in_gas_per_elec_cluster[i].push_back(vertex_is_in_gas_per_elec_cluster[j][0]);
						vertex_is_on_source_gap_per_elec_cluster[i].push_back(vertex_is_on_source_gap_per_elec_cluster[j][0]);


						
						x_start_per_elec_cluster.erase(x_start_per_elec_cluster.begin() + j);
						y_start_per_elec_cluster.erase(y_start_per_elec_cluster.begin() + j);
						z_start_per_elec_cluster.erase(z_start_per_elec_cluster.begin() + j);
						x_end_per_elec_cluster.erase(x_end_per_elec_cluster.begin() + j);
						y_end_per_elec_cluster.erase(y_end_per_elec_cluster.begin() + j);
						z_end_per_elec_cluster.erase(z_end_per_elec_cluster.begin() + j);



						x_is_on_reference_source_plane.erase(x_is_on_reference_source_plane.begin()+ j);
						y_is_on_reference_source_plane.erase(y_is_on_reference_source_plane.begin()+ j);
						z_is_on_reference_source_plane.erase(z_is_on_reference_source_plane.begin()+ j);
						x_is_on_source_foil.erase(x_is_on_source_foil.begin()+ j);
						y_is_on_source_foil.erase(y_is_on_source_foil.begin()+ j);
						z_is_on_source_foil.erase(z_is_on_source_foil.begin()+ j);
						x_is_on_main_calorimeter.erase(x_is_on_main_calorimeter.begin()+ j);
						y_is_on_main_calorimeter.erase(y_is_on_main_calorimeter.begin()+ j);
						z_is_on_main_calorimeter.erase(z_is_on_main_calorimeter.begin()+ j);
						x_is_on_x_calorimeter.erase(x_is_on_x_calorimeter.begin()+ j);
						y_is_on_x_calorimeter.erase(y_is_on_x_calorimeter.begin()+ j);
						z_is_on_x_calorimeter.erase(z_is_on_x_calorimeter.begin()+ j);
						x_is_on_gamma_veto.erase(x_is_on_gamma_veto.begin()+ j);
						y_is_on_gamma_veto.erase(y_is_on_gamma_veto.begin()+ j);
						z_is_on_gamma_veto.erase(z_is_on_gamma_veto.begin()+ j);
						x_is_on_wire.erase(x_is_on_wire.begin()+ j);
						y_is_on_wire.erase(y_is_on_wire.begin()+ j);
						z_is_on_wire.erase(z_is_on_wire.begin()+ j);
						x_is_in_gas.erase(x_is_in_gas.begin()+ j);
						y_is_in_gas.erase(y_is_in_gas.begin()+ j);
						z_is_in_gas.erase(z_is_in_gas.begin()+ j);
						x_is_on_source_gap.erase(x_is_on_source_gap.begin()+ j);
						y_is_on_source_gap.erase(y_is_on_source_gap.begin()+ j);
						z_is_on_source_gap.erase(z_is_on_source_gap.begin()+ j);

						//std::cout<<"x_start_per_elec_cluster.size()"<<x_start_per_elec_cluster.size()<<std::endl;
						//std::cout<<"x_start_per_elec_cluster[i].size()"<<x_start_per_elec_cluster[i].size()<<std::endl;

						xy_distance_from_reference_source_plane.erase(xy_distance_from_reference_source_plane.begin() + j);
						xy_distance_from_source_foil.erase(xy_distance_from_source_foil.begin() + j);
						xy_distance_from_main_calorimeter.erase(xy_distance_from_main_calorimeter.begin() + j);
						xy_distance_from_x_calorimeter.erase(xy_distance_from_x_calorimeter.begin() + j);
						xy_distance_from_gamma_veto.erase(xy_distance_from_gamma_veto.begin() + j);
						xy_distance_from_wire.erase(xy_distance_from_wire.begin() + j);
						xy_distance_from_gas.erase(xy_distance_from_gas.begin() + j);
						xy_distance_from_source_gap.erase(xy_distance_from_source_gap.begin() + j);

						xyz_distance_from_reference_source_plane.erase(xyz_distance_from_reference_source_plane.begin() + j);
						xyz_distance_from_source_foil.erase(xyz_distance_from_source_foil.begin() + j);
						xyz_distance_from_main_calorimeter.erase(xyz_distance_from_main_calorimeter.begin() + j);
						xyz_distance_from_x_calorimeter.erase(xyz_distance_from_x_calorimeter.begin() + j);
						xyz_distance_from_gamma_veto.erase(xyz_distance_from_gamma_veto.begin() + j);
						xyz_distance_from_wire.erase(xyz_distance_from_wire.begin() + j);
						xyz_distance_from_gas.erase(xyz_distance_from_gas.begin() + j);
						xyz_distance_from_source_gap.erase(xyz_distance_from_source_gap.begin() + j);

						vertex_is_on_reference_source_plane_per_elec_cluster.erase(vertex_is_on_reference_source_plane_per_elec_cluster.begin() + j);
						vertex_is_on_source_foil_per_elec_cluster.erase(vertex_is_on_source_foil_per_elec_cluster.begin() + j);
						vertex_is_on_main_calorimeter_per_elec_cluster.erase(vertex_is_on_main_calorimeter_per_elec_cluster.begin() + j);
						vertex_is_on_x_calorimeter_per_elec_cluster.erase(vertex_is_on_x_calorimeter_per_elec_cluster.begin() + j);
						vertex_is_on_gamma_veto_per_elec_cluster.erase(vertex_is_on_gamma_veto_per_elec_cluster.begin() + j);
						vertex_is_on_wire_per_elec_cluster.erase(vertex_is_on_wire_per_elec_cluster.begin() + j);
						vertex_is_in_gas_per_elec_cluster.erase(vertex_is_in_gas_per_elec_cluster.begin() + j);
						vertex_is_on_source_gap_per_elec_cluster.erase(vertex_is_on_source_gap_per_elec_cluster.begin() + j);


						//elec_in_event = false;
						ID_clsuter_per_elec_cluster.erase(ID_clsuter_per_elec_cluster.begin() + j);
						
            		} 
					else {
                		++j;
            		}
        		}
    		}
			for(int i = 0; i< x_start_per_elec_cluster.size(); i++){ // BUG HERE CAUSE IT WILL COUNT THE NAN VALUES AS A SOLUTION 
				nb_fit_solution_per_elec_cluster.push_back(x_start_per_elec_cluster[i].size());
			}
			for(int i  = 0; i< cell_num_per_elec_cluster.size(); i++ ){
				nb_cell_per_elec_cluster.push_back(cell_num_per_elec_cluster[i].size());
			}



			if(cell_num_per_elec_cluster.size() != 0){
				nb_of_elec_candidates = cell_num_per_elec_cluster.size();
				tracks_with_associated_calo = true;
			}
			

			if(DEBUG == true){                                                                                                                                                                                     
	  			std::cout<<"=========== DEBUG TRACK WITH CALO ==============="<<std::endl;                                                                                                                           
				
	  			std::cout<<"==> nb of particles with OM = "<<cell_num_per_elec_cluster.size()<<std::endl;   
				std::cout<<"nb_of_elec_candidates = "<<cell_num_per_elec_cluster.size()<<std::endl;                                                                                                         
	  			std::cout<<"DEBUG OM_HT_per_elec_cluster.size() = "<<OM_HT_per_elec_cluster.size()<<std::endl;
				std::cout<<"E_OM_per_elec_cluster.size() = "<<E_OM_per_elec_cluster.size()<<std::endl;
				std::cout<<"anode_time_per_elec_cluster.size() = "<< anode_time_per_elec_cluster.size()<<std::endl;
				for(int i = 0; i< cell_num_per_elec_cluster.size(); i++){
					std::cout<<"nb of solution = "<<x_start_per_elec_cluster[i].size()<<std::endl;
					for(int z = 0; z <x_start_per_elec_cluster[i].size(); z++ ){
					std::cout<<"Solution n° "<<z<<std::endl;	
					std::cout<<"==> START at (x , y, z) = "<<x_start_per_elec_cluster[i][z]<<", "<< y_start_per_elec_cluster[i][z]<<", " << z_start_per_elec_cluster[i][z]<<std::endl;
					std::cout<<"==> END at (x , y, z) = "<<x_end_per_elec_cluster[i][z]<<", "<< y_end_per_elec_cluster[i][z]<<", " << z_end_per_elec_cluster[i][z]<<std::endl;	
					}
				}
				
				for(int i = 0; i < cell_num_per_elec_cluster.size(); i++){
					std::cout<<"nb_fit_soluion = "<<nb_fit_solution_per_elec_cluster[i]<<std::endl;
					std::cout<<"DEBUG 2 OM_HT_per_elec_cluster[i].size() = "<<OM_HT_per_elec_cluster[i].size()<<std::endl;
	  				std::cout<<"nb of OM : "<< nb_of_OM_per_elec_cluster[i]<<std::endl;
					std::cout<<"E_OM_per_elec_cluster[i].size() = "<<E_OM_per_elec_cluster[i].size()<<std::endl;
	  			  	for(int j = 0; j < E_OM_per_elec_cluster.at(i).size(); j++){                                                                                                                                       
	  			    	std::cout<<"NUM OM = "<< OM_num_per_elec_cluster[i][j]<<" & E = "<<E_OM_per_elec_cluster[i][j]<<std::endl;
						std::cout<<"	--> Time = "<< std::setprecision(25)<<OM_timestamp_per_elec_cluster[i][j] <<std::endl;
						std::cout<<"	--> Charge = "<< std::setprecision(25)<<OM_charge_per_elec_cluster[i][j] <<std::endl;
						std::cout<<"	--> Amplitude = "<< std::setprecision(25)<<OM_amplitude_per_elec_cluster[i][j] <<std::endl;
						std::cout<<"	--> LT Only = "<< OM_LT_only_per_elec_cluster[i][j] << std::endl;
						std::cout<<"	--> HT = "<< OM_HT_per_elec_cluster[i][j] << std::endl;                                                                                         
	  			  	}
					std::cout<<"TRACK IS DELAYED : "<< elec_cluster_is_delayed[i]<<std::endl;
					std::cout<<"TRACK IS PROMPT : "<< elec_clsuter_is_prompt[i]<<std::endl;
					std::cout<<"NB OF CELLS IN TRACK = "<<cell_num_per_elec_cluster[i].size()<<std::endl;
					std::cout<<"anode_time_per_elec_cluster[i].size() = "<< anode_time_per_elec_cluster[i].size()<<std::endl;
	  			  	for(int k = 0; k <cell_num_per_elec_cluster.at(i).size(); k++){                                                                                                                                    
	  			    	std::cout<<"CELL NUM = "<< cell_num_per_elec_cluster[i][k] << std::endl;
	  			    	std::cout<<"	==> Anode time = " << std::setprecision(28) << anode_time_per_elec_cluster[i][k] << std::endl;
	  			    	std::cout<<"	==> Top cathode = "<< std::setprecision(28) << top_cathode_per_elec_cluster[i][k] <<std::endl;
	  			    	std::cout<<"	==> Bottom cathode = "<< std::setprecision(28) << bottom_cathode_per_elec_cluster[i][k]<<std::endl;
						std::cout<<"	==> z = "<< z_of_cells_per_elec_cluster[i][k]<<std::endl;
						std::cout<<"	==> sigma_z = "<< sigma_z_of_cells_per_elec_cluster[i][k]<<std::endl;
						std::cout<<"	==> r = "<<r_of_cells_per_elec_cluster[i][k]<<std::endl;
						std::cout<<"	==> sigma r = "<<sigma_r_of_cells_per_elec_cluster[i][k]<<std::endl;
						
						//std::cout<<"	==> Timestamp = "<<std::setprecision(28)<<timestamp_cell_per_elec_cluster[i][k]<<std::endl;
	  			  	}
	  			}

				  std::cout<<"elec_is_on_reference_source_plane = "<< elec_is_on_reference_source_plane<<std::endl;
				  std::cout<<"elec_is_on_source_foil = "<< elec_is_on_source_foil<<std::endl;
				  std::cout<<"elec_is_on_main_calorimeter = "<< elec_is_on_main_calorimeter<<std::endl;
				  std::cout<<"elec_is_on_x_calorimeter = "<< elec_is_on_x_calorimeter<<std::endl;
				  std::cout<<"elec_is_on_gamma_veto = "<< elec_is_on_gamma_veto<<std::endl;
				  std::cout<<"elec_is_on_wire = "<< elec_is_on_wire<<std::endl;
				  std::cout<<"elec_is_in_gas = "<< elec_is_in_gas<<std::endl;
				  std::cout<<"elec_is_on_source_gap = "<< elec_is_on_source_gap<<std::endl;
				

			}

			// END ELECTRON DETECTION
			

			// Beginin of short tracks : ==> Looping again on particles (easyer to separate each part of the code)

			for(const datatools::handle<snemo::datamodel::particle_track> & particle : PTD.particles()){
				if(particle->has_associated_calorimeter_hits() ){
					continue; // We skip electron like because already selected 
				}
				else{ // Other particles must be tracks without associated calorimeter
					if(DEBUG == true){
						std::cout<<"UND to Idenfy ! "<<std::endl;
					}
					vector<double> cell_num_of_current_track;
		    		

		    		const auto track_cluster = particle->get_trajectory_handle()->get_cluster(); // PTD -> particle_track_data.h -> particle_track.h (get_trajectory_handle()) -> tracker_trajectory.h (get_cluster()) -> tracker_cluster.h 
		    		const auto track_hits = track_cluster.hits();
		    		const auto pCD_tracker_hits = PCD.tracker_hits();
					
					vector<double> z_of_cell_for_current_track;
					vector<double> sigma_z_of_cell_for_current_track;
		    		vector<double>r_of_current_track, sigma_r_of_current_track;
					
					
					UND_cluster_is_delayed.push_back(track_cluster.is_delayed());
					UND_cluster_is_prompt.push_back(track_cluster.is_prompt());
					//std::cout<<" cluster ID = "<<track_cluster.get_cluster_id()<<std::endl;
					ID_clsuter_UND.push_back(track_cluster.get_cluster_id());
					

					// PTD Bank for tracker hits
					for(const auto hits : track_hits){// looping on all GG hit of the cluster
					 	double PTD_id = hits->get_id();

		      			cell_num_of_current_track.push_back(snemo::datamodel::gg_num(hits->get_geom_id()));

						z_of_cell_for_current_track.push_back(hits->get_z()/CLHEP::mm); // set in mm 
						sigma_z_of_cell_for_current_track.push_back(hits->get_sigma_z()/CLHEP::mm); // set in mm
						
						r_of_current_track.push_back(hits->get_r());
						sigma_r_of_current_track.push_back(hits->get_sigma_r()); 
					
					}
					//FILL information from the loop
		    		cell_num_per_UND_cluster.push_back(cell_num_of_current_track);
					z_of_cell_per_UND_cluster.push_back(z_of_cell_for_current_track);
					sigma_z_of_cells_per_UND_cluster.push_back(sigma_z_of_cell_for_current_track);
					r_of_cells_per_UND_cluster.push_back(r_of_current_track);
					sigma_r_of_cells_per_UND_cluster.push_back(sigma_r_of_current_track);					
					
					vector<double> R0_for_current_track;
	    			vector<double> Top_cathode_for_current_track;
	    			vector<double> Bottom_cathode_for_current_track;

					for(const auto pCD_track_hits : pCD_tracker_hits){
						if(std::find(cell_num_of_current_track.begin(), cell_num_of_current_track.end(), snemo::datamodel::gg_num(pCD_track_hits->get_geom_id())) != cell_num_of_current_track.end()){
							R0_for_current_track.push_back(pCD_track_hits->get_anodic_time()/CLHEP::second); // micro seconds 
							Top_cathode_for_current_track.push_back(pCD_track_hits->get_top_cathode_drift_time()/CLHEP::second);
							Bottom_cathode_for_current_track.push_back(pCD_track_hits->get_bottom_cathode_drift_time()/CLHEP::second);														
						}						
					}
					anode_time_per_UND_cluster.push_back(R0_for_current_track);		
					top_cathodes_per_UND_cluster.push_back(Top_cathode_for_current_track);
					bottom_cathodes_per_UND_cluster.push_back(Bottom_cathode_for_current_track);
					R0_for_current_track.clear();
					Top_cathode_for_current_track.clear();
					Bottom_cathode_for_current_track.clear();
					cell_num_of_current_track.clear();

					if(particle->has_trajectory()){
						x_start_per_UND_cluster.push_back({particle->get_trajectory_handle()->get_pattern().get_first()[0]});
						y_start_per_UND_cluster.push_back({particle->get_trajectory_handle()->get_pattern().get_first()[1]});
						z_start_per_UND_cluster.push_back({particle->get_trajectory_handle()->get_pattern().get_first()[2]});
						x_end_per_UND_cluster.push_back({particle->get_trajectory_handle()->get_pattern().get_last()[0]});
						y_end_per_UND_cluster.push_back({particle->get_trajectory_handle()->get_pattern().get_last()[1]});
						z_end_per_UND_cluster.push_back({particle->get_trajectory_handle()->get_pattern().get_last()[2]});
					}
					else{
						x_start_per_UND_cluster.push_back({std::nan("")});
						y_start_per_UND_cluster.push_back({std::nan("")});
						z_start_per_UND_cluster.push_back({std::nan("")});
						x_end_per_UND_cluster.push_back({std::nan("")});
						y_end_per_UND_cluster.push_back({std::nan("")});
						z_end_per_UND_cluster.push_back({std::nan("")});
					}
					if(particle->has_trajectory() && particle->has_vertices()){

						
						// Temporary vectors to store vertex data for the current particle
    					vector<double> temp_x_is_on_reference_source_plane(1, std::nan(""));
    					vector<double> temp_y_is_on_reference_source_plane(1, std::nan(""));
    					vector<double> temp_z_is_on_reference_source_plane(1, std::nan(""));
    					vector<double> temp_x_is_on_source_foil(1, std::nan(""));
    					vector<double> temp_y_is_on_source_foil(1, std::nan(""));
    					vector<double> temp_z_is_on_source_foil(1, std::nan(""));
    					vector<double> temp_x_is_on_wire(1, std::nan(""));
    					vector<double> temp_y_is_on_wire(1, std::nan(""));
    					vector<double> temp_z_is_on_wire(1, std::nan(""));
    					vector<double> temp_x_is_in_gas(1, std::nan(""));
    					vector<double> temp_y_is_in_gas(1, std::nan(""));
    					vector<double> temp_z_is_in_gas(1, std::nan(""));
    					vector<double> temp_x_is_on_source_gap(1, std::nan(""));
    					vector<double> temp_y_is_on_source_gap(1, std::nan(""));
    					vector<double> temp_z_is_on_source_gap(1, std::nan(""));
    					vector<double> temp_xy_distance_from_reference_source_plane(1, std::nan(""));
    					vector<double> temp_xy_distance_from_source_foil(1, std::nan(""));
    					vector<double> temp_xy_distance_from_wire(1, std::nan(""));
    					vector<double> temp_xy_distance_from_gas(1, std::nan(""));
    					vector<double> temp_xy_distance_from_source_gap(1, std::nan(""));
    					vector<double> temp_xyz_distance_from_reference_source_plane(1, std::nan(""));
    					vector<double> temp_xyz_distance_from_source_foil(1, std::nan(""));
    					vector<double> temp_xyz_distance_from_wire(1, std::nan(""));
    					vector<double> temp_xyz_distance_from_gas(1, std::nan(""));
    					vector<double> temp_xyz_distance_from_source_gap(1, std::nan(""));
						


						vector<double> temp_vertex_is_on_reference_source_plane_per_UND_cluster(1, 0);
						vector<double> temp_vertex_is_on_source_foil_per_UND_cluster(1, 0);
						vector<double> temp_vertex_is_on_wire_per_UND_cluster(1, 0);
						vector<double> temp_vertex_is_in_gas_per_UND_cluster(1, 0);
						vector<double> temp_vertex_is_on_source_gap_per_UND_cluster(1, 0);

						//std::cout<<"has no vertices ! :/ sadly the fucking moumou queen of gulls "<<std::endl;
						
						for(const datatools::handle<snemo::datamodel::vertex> & vertex : particle->get_vertices()){
							const geomtools::blur_spot &spot = vertex->get_spot();
							const geomtools::vector_3d &position = spot.get_position();
							
							if(vertex->is_on_reference_source_plane()){
								
								UND_is_on_reference_source_plane = true;
								temp_vertex_is_on_reference_source_plane_per_UND_cluster[0] = 1;
								temp_x_is_on_reference_source_plane[0] = position[0];
								temp_y_is_on_reference_source_plane[0] = position[1];
								temp_z_is_on_reference_source_plane[0] = position[2];
								temp_xy_distance_from_reference_source_plane[0] = vertex->get_distance_xy();
								temp_xyz_distance_from_reference_source_plane[0] = vertex->get_distance();
							}
							
							if(vertex->is_on_source_foil()){
								
								UND_is_on_source_foil = true;
								temp_vertex_is_on_source_foil_per_UND_cluster[0] = 1;
								temp_x_is_on_source_foil[0] =  position[0];
								temp_y_is_on_source_foil[0] =  position[1];
								temp_z_is_on_source_foil[0] =  position[2];
								temp_xy_distance_from_source_foil[0] =  vertex->get_distance_xy();
								temp_xyz_distance_from_source_foil[0] =  vertex->get_distance();
							}
							
							if(vertex->is_on_wire()){
								
								UND_is_on_wire = true;
								temp_vertex_is_on_wire_per_UND_cluster[0] = 1;
								temp_x_is_on_wire[0] = position[0];
								temp_y_is_on_wire[0] = position[1];
								temp_z_is_on_wire[0] = position[2];
								temp_xy_distance_from_wire[0] =  vertex->get_distance_xy();
								temp_xyz_distance_from_wire[0] = vertex->get_distance();
							}
							
							if(vertex->is_in_gas()){
								
								UND_is_in_gas = true;
								temp_vertex_is_in_gas_per_UND_cluster[0] = 1;
								temp_x_is_in_gas[0] = position[0];
								temp_y_is_in_gas[0] = position[1];
								temp_z_is_in_gas[0] = position[2];
								temp_xy_distance_from_gas[0] = vertex->get_distance_xy();
								temp_xyz_distance_from_gas[0] = vertex->get_distance();
							}
							
							if(vertex->is_on_source_gap()){
								
								UND_is_on_source_gap = true;
								temp_vertex_is_on_source_gap_per_UND_cluster[0] = 1;
								temp_x_is_on_source_gap[0] = position[0];
								temp_y_is_on_source_gap[0] = position[1];
								temp_z_is_on_source_gap[0] = position[2];
								temp_xy_distance_from_source_gap[0] = vertex->get_distance_xy();
								temp_xyz_distance_from_source_gap[0] = vertex->get_distance();
							}
						
						}
						// Push the collected data for this particle into the main vectors
    					x_is_on_reference_source_plane_per_UND_cluster.push_back(temp_x_is_on_reference_source_plane);
    					y_is_on_reference_source_plane_per_UND_cluster.push_back(temp_y_is_on_reference_source_plane);
    					z_is_on_reference_source_plane_per_UND_cluster.push_back(temp_z_is_on_reference_source_plane);
    					x_is_on_source_foil_per_UND_cluster.push_back(temp_x_is_on_source_foil);
    					y_is_on_source_foil_per_UND_cluster.push_back(temp_y_is_on_source_foil);
    					z_is_on_source_foil_per_UND_cluster.push_back(temp_z_is_on_source_foil);
    					x_is_on_wire_per_UND_cluster.push_back(temp_x_is_on_wire);
    					y_is_on_wire_per_UND_cluster.push_back(temp_y_is_on_wire);
    					z_is_on_wire_per_UND_cluster.push_back(temp_z_is_on_wire);
    					x_is_in_gas_per_UND_cluster.push_back(temp_x_is_in_gas);
    					y_is_in_gas_per_UND_cluster.push_back(temp_y_is_in_gas);
    					z_is_in_gas_per_UND_cluster.push_back(temp_z_is_in_gas);
    					x_is_on_source_gap_per_UND_cluster.push_back(temp_x_is_on_source_gap);
    					y_is_on_source_gap_per_UND_cluster.push_back(temp_y_is_on_source_gap);
    					z_is_on_source_gap_per_UND_cluster.push_back(temp_z_is_on_source_gap);
    					xy_distance_from_reference_source_plane_per_UND_cluster.push_back(temp_xy_distance_from_reference_source_plane);
    					xy_distance_from_source_foil_per_UND_cluster.push_back(temp_xy_distance_from_source_foil);
    					xy_distance_from_wire_per_UND_cluster.push_back(temp_xy_distance_from_wire);
    					xy_distance_from_gas_per_UND_cluster.push_back(temp_xy_distance_from_gas);
    					xy_distance_from_source_gap_per_UND_cluster.push_back(temp_xy_distance_from_source_gap);
    					xyz_distance_from_reference_source_plane_per_UND_cluster.push_back(temp_xyz_distance_from_reference_source_plane);
    					xyz_distance_from_source_foil_per_UND_cluster.push_back(temp_xyz_distance_from_source_foil);
    					xyz_distance_from_wire_per_UND_cluster.push_back(temp_xyz_distance_from_wire);
    					xyz_distance_from_gas_per_UND_cluster.push_back(temp_xyz_distance_from_gas);
    					xyz_distance_from_source_gap_per_UND_cluster.push_back(temp_xyz_distance_from_source_gap);
						
						
						vertex_is_on_reference_source_plane_per_UND_cluster.push_back(temp_vertex_is_on_reference_source_plane_per_UND_cluster);
						vertex_is_on_source_foil_per_UND_cluster.push_back(temp_vertex_is_on_source_foil_per_UND_cluster);
						vertex_is_on_wire_per_UND_cluster.push_back(temp_vertex_is_on_wire_per_UND_cluster);
						vertex_is_in_gas_per_UND_cluster.push_back(temp_vertex_is_in_gas_per_UND_cluster);
						vertex_is_on_source_gap_per_UND_cluster.push_back(temp_vertex_is_on_source_gap_per_UND_cluster);

					}
					else{
						
						
						x_is_on_reference_source_plane_per_UND_cluster.push_back({std::nan("")});
						y_is_on_reference_source_plane_per_UND_cluster.push_back({std::nan("")});
						z_is_on_reference_source_plane_per_UND_cluster.push_back({std::nan("")});
						xy_distance_from_reference_source_plane_per_UND_cluster.push_back({std::nan("")});
						xyz_distance_from_reference_source_plane_per_UND_cluster.push_back({std::nan("")});
						
						x_is_on_source_foil_per_UND_cluster.push_back({std::nan("")});
						y_is_on_source_foil_per_UND_cluster.push_back({std::nan("")});
						z_is_on_source_foil_per_UND_cluster.push_back({std::nan("")});
						xy_distance_from_source_foil_per_UND_cluster.push_back({std::nan("")});
						xyz_distance_from_source_foil_per_UND_cluster.push_back({std::nan("")});
						
						x_is_on_wire_per_UND_cluster.push_back({std::nan("")});
						y_is_on_wire_per_UND_cluster.push_back({std::nan("")});
						z_is_on_wire_per_UND_cluster.push_back({std::nan("")});
						xy_distance_from_wire_per_UND_cluster.push_back({std::nan("")});
						xyz_distance_from_wire_per_UND_cluster.push_back({std::nan("")});
						
						x_is_in_gas_per_UND_cluster.push_back({std::nan("")});
						y_is_in_gas_per_UND_cluster.push_back({std::nan("")});
						z_is_in_gas_per_UND_cluster.push_back({std::nan("")});
						xy_distance_from_gas_per_UND_cluster.push_back({std::nan("")});
						xyz_distance_from_gas_per_UND_cluster.push_back({std::nan("")});
						
						x_is_on_source_gap_per_UND_cluster.push_back({std::nan("")});
						y_is_on_source_gap_per_UND_cluster.push_back({std::nan("")});
						z_is_on_source_gap_per_UND_cluster.push_back({std::nan("")});
						xy_distance_from_source_gap_per_UND_cluster.push_back({std::nan("")});
						xyz_distance_from_source_gap_per_UND_cluster.push_back({std::nan("")});	

						vertex_is_on_reference_source_plane_per_UND_cluster.push_back({0});
						vertex_is_on_source_foil_per_UND_cluster.push_back({0});
						vertex_is_on_wire_per_UND_cluster.push_back({0});
						vertex_is_in_gas_per_UND_cluster.push_back({0});
						vertex_is_on_source_gap_per_UND_cluster.push_back({0});

					}
				}
			}
			//std::cout<<"Chicken = "<<cell_num_per_UND_cluster.size()<<std::endl;
			//std::cout<<"POULET = "<<x_is_on_reference_source_plane_per_UND_cluster.size()<<std::endl;
			//std::cout<<"DINDE = "<<x_is_on_reference_source_plane_per_UND_cluster[0].size()<<std::endl;
			
			// Check if 2 particles hase the same cluster, if yes, erase one
			//std::cout<<" HERE ! ==> cell_num_per_UND_cluster.size() = "<<cell_num_per_UND_cluster.size()<<std::endl;

			for (size_t i = 0; i < cell_num_per_UND_cluster.size(); ++i) {
        		for (size_t j = i + 1; j < cell_num_per_UND_cluster.size(); ) {
            		if (cell_num_per_UND_cluster[i]== cell_num_per_UND_cluster[j] ) {
                		
                		// Remove duplicate
						
                		cell_num_per_UND_cluster.erase(cell_num_per_UND_cluster.begin() + j);
  						anode_time_per_UND_cluster.erase(anode_time_per_UND_cluster.begin() + j);
  						top_cathodes_per_UND_cluster.erase(top_cathodes_per_UND_cluster.begin() + j);
  						bottom_cathodes_per_UND_cluster.erase(bottom_cathodes_per_UND_cluster.begin() + j);
						z_of_cell_per_UND_cluster.erase(z_of_cell_per_UND_cluster.begin() + j);
						sigma_z_of_cells_per_UND_cluster.erase(sigma_z_of_cells_per_UND_cluster.begin() + j);
						r_of_cells_per_UND_cluster.erase(r_of_cells_per_UND_cluster.begin() + j);
						sigma_r_of_cells_per_UND_cluster.erase(sigma_r_of_cells_per_UND_cluster.begin() + j);
						UND_cluster_is_delayed.erase(UND_cluster_is_delayed.begin() + j);
						UND_cluster_is_prompt.erase(UND_cluster_is_prompt.begin() + j);
						
						
						x_start_per_UND_cluster[i].push_back(x_start_per_UND_cluster[j][0]);
						y_start_per_UND_cluster[i].push_back(y_start_per_UND_cluster[j][0]);
						z_start_per_UND_cluster[i].push_back(z_start_per_UND_cluster[j][0]);
						x_end_per_UND_cluster[i].push_back(x_end_per_UND_cluster[j][0]);
						y_end_per_UND_cluster[i].push_back(y_end_per_UND_cluster[j][0]);
						z_end_per_UND_cluster[i].push_back(z_end_per_UND_cluster[j][0]);
						

						x_is_on_reference_source_plane_per_UND_cluster[i].push_back(x_is_on_reference_source_plane_per_UND_cluster[j][0]);
						
						y_is_on_reference_source_plane_per_UND_cluster[i].push_back(y_is_on_reference_source_plane_per_UND_cluster[j][0]);
						z_is_on_reference_source_plane_per_UND_cluster[i].push_back(z_is_on_reference_source_plane_per_UND_cluster[j][0]);
						xy_distance_from_reference_source_plane_per_UND_cluster[i].push_back(xy_distance_from_reference_source_plane_per_UND_cluster[j][0]);
						xyz_distance_from_reference_source_plane_per_UND_cluster[i].push_back(xyz_distance_from_reference_source_plane_per_UND_cluster[j][0]);
						
						x_is_on_source_foil_per_UND_cluster[i].push_back(x_is_on_source_foil_per_UND_cluster[j][0]);
						y_is_on_source_foil_per_UND_cluster[i].push_back(y_is_on_source_foil_per_UND_cluster[j][0]);
						z_is_on_source_foil_per_UND_cluster[i].push_back(z_is_on_source_foil_per_UND_cluster[j][0]);
						xy_distance_from_source_foil_per_UND_cluster[i].push_back(xy_distance_from_source_foil_per_UND_cluster[j][0]);
						xyz_distance_from_source_foil_per_UND_cluster[i].push_back(xyz_distance_from_source_foil_per_UND_cluster[j][0]);
						x_is_on_wire_per_UND_cluster[i].push_back(x_is_on_wire_per_UND_cluster[j][0]);
						
						y_is_on_wire_per_UND_cluster[i].push_back(y_is_on_wire_per_UND_cluster[j][0]);
						
						z_is_on_wire_per_UND_cluster[i].push_back(z_is_on_wire_per_UND_cluster[j][0]);
						
						xy_distance_from_wire_per_UND_cluster[i].push_back(xy_distance_from_wire_per_UND_cluster[j][0]);
						
						xyz_distance_from_wire_per_UND_cluster[i].push_back(xyz_distance_from_wire_per_UND_cluster[j][0]);
						
						x_is_in_gas_per_UND_cluster[i].push_back(x_is_in_gas_per_UND_cluster[j][0]);
						y_is_in_gas_per_UND_cluster[i].push_back(y_is_in_gas_per_UND_cluster[j][0]);
						z_is_in_gas_per_UND_cluster[i].push_back(z_is_in_gas_per_UND_cluster[j][0]);
						xy_distance_from_gas_per_UND_cluster[i].push_back(xy_distance_from_gas_per_UND_cluster[j][0]);
						xyz_distance_from_gas_per_UND_cluster[i].push_back(xyz_distance_from_gas_per_UND_cluster[j][0]);
						x_is_on_source_gap_per_UND_cluster[i].push_back(x_is_on_source_gap_per_UND_cluster[j][0]);
						y_is_on_source_gap_per_UND_cluster[i].push_back(y_is_on_source_gap_per_UND_cluster[j][0]);
						z_is_on_source_gap_per_UND_cluster[i].push_back(z_is_on_source_gap_per_UND_cluster[j][0]);
						xy_distance_from_source_gap_per_UND_cluster[i].push_back(xy_distance_from_source_gap_per_UND_cluster[j][0]);
						xyz_distance_from_source_gap_per_UND_cluster[i].push_back(xyz_distance_from_source_gap_per_UND_cluster[j][0]);
						

						vertex_is_on_reference_source_plane_per_UND_cluster[i].push_back(vertex_is_on_reference_source_plane_per_UND_cluster[j][0]);
						vertex_is_on_source_foil_per_UND_cluster[i].push_back(vertex_is_on_source_foil_per_UND_cluster[j][0]);
						vertex_is_on_wire_per_UND_cluster[i].push_back(vertex_is_on_wire_per_UND_cluster[j][0]);
						vertex_is_in_gas_per_UND_cluster[i].push_back(vertex_is_in_gas_per_UND_cluster[j][0]);
						vertex_is_on_source_gap_per_UND_cluster[i].push_back(vertex_is_on_source_gap_per_UND_cluster[j][0]);


						x_is_on_reference_source_plane_per_UND_cluster.erase(x_is_on_reference_source_plane_per_UND_cluster.begin() + j);
						y_is_on_reference_source_plane_per_UND_cluster.erase(y_is_on_reference_source_plane_per_UND_cluster.begin() + j);
						z_is_on_reference_source_plane_per_UND_cluster.erase(z_is_on_reference_source_plane_per_UND_cluster.begin() + j);
						xy_distance_from_reference_source_plane_per_UND_cluster.erase(xy_distance_from_reference_source_plane_per_UND_cluster.begin() + j);
						xyz_distance_from_reference_source_plane_per_UND_cluster.erase(xyz_distance_from_reference_source_plane_per_UND_cluster.begin() + j);
						x_is_on_source_foil_per_UND_cluster.erase(x_is_on_source_foil_per_UND_cluster.begin() + j);
						y_is_on_source_foil_per_UND_cluster.erase(y_is_on_source_foil_per_UND_cluster.begin() + j);
						z_is_on_source_foil_per_UND_cluster.erase(z_is_on_source_foil_per_UND_cluster.begin() + j);
						xy_distance_from_source_foil_per_UND_cluster.erase(xy_distance_from_source_foil_per_UND_cluster.begin() + j);
						xyz_distance_from_source_foil_per_UND_cluster.erase(xyz_distance_from_source_foil_per_UND_cluster.begin() + j);
						x_is_on_wire_per_UND_cluster.erase(x_is_on_wire_per_UND_cluster.begin() + j);
						y_is_on_wire_per_UND_cluster.erase(y_is_on_wire_per_UND_cluster.begin() + j);
						z_is_on_wire_per_UND_cluster.erase(z_is_on_wire_per_UND_cluster.begin() + j);
						xy_distance_from_wire_per_UND_cluster.erase(xy_distance_from_wire_per_UND_cluster.begin() + j);
						xyz_distance_from_wire_per_UND_cluster.erase(xyz_distance_from_wire_per_UND_cluster.begin() + j);
						x_is_in_gas_per_UND_cluster.erase(x_is_in_gas_per_UND_cluster.begin() + j);
						y_is_in_gas_per_UND_cluster.erase(y_is_in_gas_per_UND_cluster.begin() + j);
						z_is_in_gas_per_UND_cluster.erase(z_is_in_gas_per_UND_cluster.begin() + j);
						xy_distance_from_gas_per_UND_cluster.erase(xy_distance_from_gas_per_UND_cluster.begin() + j);
						xyz_distance_from_gas_per_UND_cluster.erase(xyz_distance_from_gas_per_UND_cluster.begin() + j);
						x_is_on_source_gap_per_UND_cluster.erase(x_is_on_source_gap_per_UND_cluster.begin() + j);
						y_is_on_source_gap_per_UND_cluster.erase(y_is_on_source_gap_per_UND_cluster.begin() + j);
						z_is_on_source_gap_per_UND_cluster.erase(z_is_on_source_gap_per_UND_cluster.begin() + j);
						xy_distance_from_source_gap_per_UND_cluster.erase(xy_distance_from_source_gap_per_UND_cluster.begin() + j);
						xyz_distance_from_source_gap_per_UND_cluster.erase(xyz_distance_from_source_gap_per_UND_cluster.begin() + j);
						
						x_start_per_UND_cluster.erase(x_start_per_UND_cluster.begin() + j);
						y_start_per_UND_cluster.erase(y_start_per_UND_cluster.begin() + j);
						z_start_per_UND_cluster.erase(z_start_per_UND_cluster.begin() + j);
						x_end_per_UND_cluster.erase(x_end_per_UND_cluster.begin() + j);
						y_end_per_UND_cluster.erase(y_end_per_UND_cluster.begin() + j);
						z_end_per_UND_cluster.erase(z_end_per_UND_cluster.begin() + j);
						
						ID_clsuter_UND.erase(ID_clsuter_UND.begin() + j);

						vertex_is_on_reference_source_plane_per_UND_cluster.erase(vertex_is_on_reference_source_plane_per_UND_cluster.begin() + j);
						vertex_is_on_source_foil_per_UND_cluster.erase(vertex_is_on_source_foil_per_UND_cluster.begin() + j);
						vertex_is_on_wire_per_UND_cluster.erase(vertex_is_on_wire_per_UND_cluster.begin() + j);
						vertex_is_in_gas_per_UND_cluster.erase(vertex_is_in_gas_per_UND_cluster.begin() + j);
						vertex_is_on_source_gap_per_UND_cluster.erase(vertex_is_on_source_gap_per_UND_cluster.begin() + j);
            		}
					else {
                		++j;
            		}
        		}
    		}
			
			// Modification on the code on 18/06/2025
			// Here we need to remove all UND cluster that have the same cluster ID of electron cluster (==> To remove particles with 2 fit solutions i.e one with extrapolated vertex in MW with calo hit, the other with no calo hit)
			for (size_t i = 0; i < cell_num_per_elec_cluster.size(); ++i){
				for(size_t j = 0; j< cell_num_per_UND_cluster.size(); ++j){
					if(ID_clsuter_per_elec_cluster[i] == ID_clsuter_UND[j]){ // If ID of elec cluster is the same as UND cluster, we remove the UND cluster cause it's abviously an electron 
						 
						cell_num_per_UND_cluster.erase(cell_num_per_UND_cluster.begin() + j);
  						anode_time_per_UND_cluster.erase(anode_time_per_UND_cluster.begin() + j);
  						top_cathodes_per_UND_cluster.erase(top_cathodes_per_UND_cluster.begin() + j);
  						bottom_cathodes_per_UND_cluster.erase(bottom_cathodes_per_UND_cluster.begin() + j);
						z_of_cell_per_UND_cluster.erase(z_of_cell_per_UND_cluster.begin() + j);
						sigma_z_of_cells_per_UND_cluster.erase(sigma_z_of_cells_per_UND_cluster.begin() + j);
						r_of_cells_per_UND_cluster.erase(r_of_cells_per_UND_cluster.begin() + j);
						sigma_r_of_cells_per_UND_cluster.erase(sigma_r_of_cells_per_UND_cluster.begin() + j);
						UND_cluster_is_delayed.erase(UND_cluster_is_delayed.begin() + j);
						UND_cluster_is_prompt.erase(UND_cluster_is_prompt.begin() + j);
						
						
						


						x_is_on_reference_source_plane_per_UND_cluster.erase(x_is_on_reference_source_plane_per_UND_cluster.begin() + j);
						y_is_on_reference_source_plane_per_UND_cluster.erase(y_is_on_reference_source_plane_per_UND_cluster.begin() + j);
						z_is_on_reference_source_plane_per_UND_cluster.erase(z_is_on_reference_source_plane_per_UND_cluster.begin() + j);
						xy_distance_from_reference_source_plane_per_UND_cluster.erase(xy_distance_from_reference_source_plane_per_UND_cluster.begin() + j);
						xyz_distance_from_reference_source_plane_per_UND_cluster.erase(xyz_distance_from_reference_source_plane_per_UND_cluster.begin() + j);
						x_is_on_source_foil_per_UND_cluster.erase(x_is_on_source_foil_per_UND_cluster.begin() + j);
						y_is_on_source_foil_per_UND_cluster.erase(y_is_on_source_foil_per_UND_cluster.begin() + j);
						z_is_on_source_foil_per_UND_cluster.erase(z_is_on_source_foil_per_UND_cluster.begin() + j);
						xy_distance_from_source_foil_per_UND_cluster.erase(xy_distance_from_source_foil_per_UND_cluster.begin() + j);
						xyz_distance_from_source_foil_per_UND_cluster.erase(xyz_distance_from_source_foil_per_UND_cluster.begin() + j);
						x_is_on_wire_per_UND_cluster.erase(x_is_on_wire_per_UND_cluster.begin() + j);
						y_is_on_wire_per_UND_cluster.erase(y_is_on_wire_per_UND_cluster.begin() + j);
						z_is_on_wire_per_UND_cluster.erase(z_is_on_wire_per_UND_cluster.begin() + j);
						xy_distance_from_wire_per_UND_cluster.erase(xy_distance_from_wire_per_UND_cluster.begin() + j);
						xyz_distance_from_wire_per_UND_cluster.erase(xyz_distance_from_wire_per_UND_cluster.begin() + j);
						x_is_in_gas_per_UND_cluster.erase(x_is_in_gas_per_UND_cluster.begin() + j);
						y_is_in_gas_per_UND_cluster.erase(y_is_in_gas_per_UND_cluster.begin() + j);
						z_is_in_gas_per_UND_cluster.erase(z_is_in_gas_per_UND_cluster.begin() + j);
						xy_distance_from_gas_per_UND_cluster.erase(xy_distance_from_gas_per_UND_cluster.begin() + j);
						xyz_distance_from_gas_per_UND_cluster.erase(xyz_distance_from_gas_per_UND_cluster.begin() + j);
						x_is_on_source_gap_per_UND_cluster.erase(x_is_on_source_gap_per_UND_cluster.begin() + j);
						y_is_on_source_gap_per_UND_cluster.erase(y_is_on_source_gap_per_UND_cluster.begin() + j);
						z_is_on_source_gap_per_UND_cluster.erase(z_is_on_source_gap_per_UND_cluster.begin() + j);
						xy_distance_from_source_gap_per_UND_cluster.erase(xy_distance_from_source_gap_per_UND_cluster.begin() + j);
						xyz_distance_from_source_gap_per_UND_cluster.erase(xyz_distance_from_source_gap_per_UND_cluster.begin() + j);
						
						x_start_per_UND_cluster.erase(x_start_per_UND_cluster.begin() + j);
						y_start_per_UND_cluster.erase(y_start_per_UND_cluster.begin() + j);
						z_start_per_UND_cluster.erase(z_start_per_UND_cluster.begin() + j);
						x_end_per_UND_cluster.erase(x_end_per_UND_cluster.begin() + j);
						y_end_per_UND_cluster.erase(y_end_per_UND_cluster.begin() + j);
						z_end_per_UND_cluster.erase(z_end_per_UND_cluster.begin() + j);
						
						ID_clsuter_UND.erase(ID_clsuter_UND.begin() + j);

						vertex_is_on_reference_source_plane_per_UND_cluster.erase(vertex_is_on_reference_source_plane_per_UND_cluster.begin() + j);
						vertex_is_on_source_foil_per_UND_cluster.erase(vertex_is_on_source_foil_per_UND_cluster.begin() + j);
						vertex_is_on_wire_per_UND_cluster.erase(vertex_is_on_wire_per_UND_cluster.begin() + j);
						vertex_is_in_gas_per_UND_cluster.erase(vertex_is_in_gas_per_UND_cluster.begin() + j);
						vertex_is_on_source_gap_per_UND_cluster.erase(vertex_is_on_source_gap_per_UND_cluster.begin() + j);


					}
				}
			}

			// Find min R0 timestamp for each UND particle

			for(int i = 0; i < cell_num_per_UND_cluster.size(); i++){
				//min_time_per_UND_cluster.push_back(*min_element(anode_time_per_UND_cluster[i].begin(), anode_time_per_UND_cluster[i].end())); // anode time already in micro seconds
				
				//									MEAN VALUE Just in case... (:) 
				
				double sum = std::accumulate(anode_time_per_UND_cluster[i].begin(), anode_time_per_UND_cluster[i].end(), 0.0);
				double mean = sum / anode_time_per_UND_cluster[i].size();
				min_time_per_UND_cluster.push_back(mean);
			}
			for(int i = 0; i< x_start_per_UND_cluster.size(); i++){ 
				nb_fit_solution_per_UND_cluster.push_back(x_start_per_UND_cluster[i].size());
			}

			for(int i = 0; i < cell_num_per_UND_cluster.size(); i++){
				nb_cell_per_UND_cluster.push_back(cell_num_per_UND_cluster[i].size());
			}
			if(cell_num_per_UND_cluster.size() != 0){
				nb_of_UND_candidates = cell_num_per_UND_cluster.size();
				tracks_without_associated_calo = true;
			}







			if(DEBUG == true){
				std::cout<<"\n\n"<<std::endl;
				std::cout<<"======================  UNASOCIATED TRACKS ======================\n\n"<<std::endl;
				
				std::cout<<"nb_of_UND_candidates : "<<nb_of_UND_candidates<<std::endl;
				for(int i = 0; i < cell_num_per_UND_cluster.size(); ++i){
					std::cout<<"	==> ID cluster (particle ? ) = "<<ID_clsuter_UND[i]<<std::endl;
					for(int j = 0; j< cell_num_per_UND_cluster[i].size(); ++j){
						std::cout<<"		==> NUM CELL = "<<cell_num_per_UND_cluster[i][j]<<std::endl;
						std::cout<<"			R0 = "<<anode_time_per_UND_cluster[i][j] <<std::endl;
						std::cout<<"			R5 = "<<top_cathodes_per_UND_cluster[i][j] <<std::endl;
						std::cout<<"			R6 = "<<bottom_cathodes_per_UND_cluster[i][j] <<std::endl;

						std::cout<<std::endl;
					}
					for(int j = 0; j < x_start_per_UND_cluster[i].size(); j++){
						std::cout<<"			==> Solution : "<< j<<std::endl;
						std::cout<<"				==> START at (x , y, z) = "<<x_start_per_UND_cluster[i][j]<<", "<< y_start_per_UND_cluster[i][j]<<", " << z_start_per_UND_cluster[i][j]<<std::endl;
						std::cout<<"				==> END at (x , y, z) = "<<x_end_per_UND_cluster[i][j]<<", "<< y_end_per_UND_cluster[i][j]<<", " << z_end_per_UND_cluster[i][j]<<std::endl;	
					}


				}
			}

      	}

		
		


		//std::cout<<"\n========================\n"<<std::endl;

		if(evt_isolated_calo == 1 && DEBUG == true){
			std::cout<<"Nb de gamma = "<< nb_isolated_calo<<std::endl;
		}

		if(tracks_with_associated_calo ==true && DEBUG == true) {
            for (int j = 0; j < nb_of_elec_candidates; j++) {
                std::cout << "Electron Candidate " << j << " : evt = "<< event_number << std::endl;
                std::cout << "Number of cells: " << cell_num_per_elec_cluster.size() << std::endl;
                std::cout << "Number of OM: " << OM_num_per_elec_cluster.size() << std::endl;
                std::cout << "Number of OM LT only per cluster: " << OM_LT_only_per_elec_cluster.size() << std::endl;
				std::cout << "Number of OM LT only per cluster at "<< j << " ==> "<< OM_LT_only_per_elec_cluster.at(j).size()<< std::endl;
				for(int g = 0; g < OM_LT_only_per_elec_cluster.at(j).size(); g++){
					std::cout<<"	==> LT : "<< OM_LT_only_per_elec_cluster.at(j).at(g) << std::endl;
					std::cout<<"	==> HT : "<< OM_HT_per_elec_cluster.at(j).at(g) << std::endl;
					
				}


				std::cout<<"nb_fit_solution_per_elec_cluster[j] = "<<nb_fit_solution_per_elec_cluster[j]<<std::endl;
				std::cout<<"x_is_on_source_foil.size() = "<<x_is_on_source_foil.size()<< std::endl;
				for(int f = 0 ; f < nb_fit_solution_per_elec_cluster[j]; f++){
					std::cout<<"----"<<std::endl;
					std::cout<<"x_is_on_source_foil[i].size() = "<< x_is_on_source_foil[j].size()<<std::endl;
					std::cout<<"LA ! "<<std::endl;
					
					
					std::cout<<"x_is_on_source_foil[i][0] = "<<x_is_on_source_foil[j][f]<<std::endl;
					std::cout<<"x_is_on_reference_source_plane[i][0] = "<< x_is_on_reference_source_plane[j][f] <<std::endl;
					std::cout<<"x_is_on_main_calorimeter[i][0] = "<< x_is_on_main_calorimeter[j][f] <<std::endl;
					std::cout<<"x_is_on_x_calorimeter[i][0] = "<< x_is_on_x_calorimeter[j][f]<<std::endl;
					std::cout<<"x_is_on_gamma_veto[i][0] = "<< x_is_on_gamma_veto[j][f]<<std::endl;
					std::cout<<"x_is_on_wire[i][0] = "<<x_is_on_wire[j][f] <<std::endl;
					std::cout<<"x_is_in_gas[i][0] = "<< x_is_in_gas[j][f]<<std::endl;
					std::cout<<"x_is_on_source_gap[i][0] = "<< x_is_on_source_gap[j][f]<<std::endl;

					std::cout<<"	==> all y : "<<std::endl;

					std::cout<<"y_is_on_source_foil[i][0] = "<<y_is_on_source_foil[j][f]<<std::endl;
					std::cout<<"y_is_on_reference_source_plane[i][0] = "<<y_is_on_reference_source_plane[j][f] <<std::endl;
					std::cout<<"y_is_on_main_calorimeter[i][0] = "<<y_is_on_main_calorimeter[j][f] <<std::endl;
					std::cout<<"y_is_on_x_calorimeter[i][0] = "<<y_is_on_x_calorimeter[j][f]<<std::endl;
					std::cout<<"y_is_on_gamma_veto[i][0] = "<<y_is_on_gamma_veto[j][f]<<std::endl;
					std::cout<<"y_is_on_wire[i][0] = "<<y_is_on_wire[j][f] <<std::endl;
					std::cout<<"y_is_in_gas[i][0] = "<<y_is_in_gas[j][f]<<std::endl;
					std::cout<<"y_is_on_source_gap[i][0] = "<<y_is_on_source_gap[j][f]<<std::endl;

					std::cout<<"\nEND ; "<<std::endl;
				}
						

                std::cout << "Number of OM HT per cluster: " << OM_HT_per_elec_cluster.size() << std::endl;
                std::cout << "Number of OM timestamp per cluster: " << OM_timestamp_per_elec_cluster.size() << std::endl;
                std::cout << "Number of OM charge per cluster: " << OM_charge_per_elec_cluster.size() << std::endl;
                std::cout << "Number of OM amplitude per cluster: " << OM_amplitude_per_elec_cluster.size() << std::endl;
                std::cout << "Number of z of cells per cluster: " << z_of_cells_per_elec_cluster.size() << std::endl;
                std::cout << "Number of sigma z of cells per cluster: " << sigma_z_of_cells_per_elec_cluster.size() << std::endl;
                std::cout << "Number of r of cells per cluster: " << r_of_cells_per_elec_cluster.size() << std::endl;
                std::cout << "Number of sigma r of cells per cluster: " << sigma_r_of_cells_per_elec_cluster.size() << std::endl;
                std::cout << "Number of x start per cluster: " << x_start_per_elec_cluster.size() << std::endl;
                std::cout << "Number of y start per cluster: " << y_start_per_elec_cluster.size() << std::endl;
                std::cout << "Number of z start per cluster: " << z_start_per_elec_cluster.size() << std::endl;
                std::cout << "Number of x end per cluster: " << x_end_per_elec_cluster.size() << std::endl;
                std::cout << "Number of y end per cluster: " << y_end_per_elec_cluster.size() << std::endl;
                std::cout << "Number of z end per cluster: " << z_end_per_elec_cluster.size() << std::endl;
                std::cout << "Number of ID cluster per cluster: " << ID_clsuter_per_elec_cluster[j] << std::endl;
                std::cout << "Number of x is on reference source plane: " << x_is_on_reference_source_plane.size() << std::endl;
                std::cout << "Number of y is on reference source plane: " << y_is_on_reference_source_plane.size() << std::endl;
                std::cout << "Number of z is on reference source plane: " << z_is_on_reference_source_plane.size() << std::endl;
                std::cout << "Number of x is on source foil: " << x_is_on_source_foil.size() << std::endl;
                std::cout << "Number of y is on source foil: " << y_is_on_source_foil.size() << std::endl;
                std::cout << "Number of z is on source foil: " << z_is_on_source_foil.size() << std::endl;
                std::cout << "Number of x is on main calorimeter: " << x_is_on_main_calorimeter.size() << std::endl;
                std::cout << "Number of y is on main calorimeter: " << y_is_on_main_calorimeter.size() << std::endl;
                std::cout << "Number of z is on main calorimeter: " << z_is_on_main_calorimeter.size() << std::endl;
                std::cout << "Number of x is on x calorimeter: " << x_is_on_x_calorimeter.size() << std::endl;
                std::cout << "Number of y is on x calorimeter: " << y_is_on_x_calorimeter.size() << std::endl;
                std::cout << "Number of z is on x calorimeter: " << z_is_on_x_calorimeter.size() << std::endl;
                std::cout << "Number of x is on gamma veto: " << x_is_on_gamma_veto.size() << std::endl;
                std::cout << "Number of y is on gamma veto: " << y_is_on_gamma_veto.size() << std::endl;
                std::cout << "Number of z is on gamma veto: " << z_is_on_gamma_veto.size() << std::endl;
                std::cout << "Number of x is on wire: " << x_is_on_wire.size() << std::endl;
                std::cout << "Number of y is on wire: " << y_is_on_wire.size() << std::endl;
                std::cout << "Number of z is on wire: " << z_is_on_wire.size() << std::endl;
                std::cout << "Number of x is in gas: " << x_is_in_gas.size() << std::endl;
                std::cout << "Number of y is in gas: " << y_is_in_gas.size() << std::endl;
                std::cout << "Number of z is in gas: " << z_is_in_gas.size() << std::endl;
                std::cout << "Number of x is on source gap: " << x_is_on_source_gap.size() << std::endl;
                std::cout << "Number of y is on source gap: " << y_is_on_source_gap.size() << std::endl;
                std::cout << "Number of z is on source gap: " << z_is_on_source_gap.size() << std::endl;
                std::cout << "Number of xy distance from reference source plane: " << xy_distance_from_reference_source_plane.size() << std::endl;
                std::cout << "Number of xy distance from source foil: " << xy_distance_from_source_foil.size() << std::endl;
                std::cout << "Number of xy distance from main calorimeter: " << xy_distance_from_main_calorimeter.size() << std::endl;

                std::cout << "Number of xyz distance from reference source plane: " << xyz_distance_from_reference_source_plane.size() << std::endl;
                std::cout << "Number of xyz distance from source foil: " << xyz_distance_from_source_foil.size() << std::endl;
                std::cout << "Number of xyz distance from main calorimeter: " << xyz_distance_from_main_calorimeter.size() << std::endl;
                std::cout << "Number of xyz distance from x calorimeter: " << xyz_distance_from_x_calorimeter.size() << std::endl;
                std::cout << "Number of xyz distance from gamma veto: " << xyz_distance_from_gamma_veto.size() << std::endl;
                std::cout << "Number of xyz distance from wire: " << xyz_distance_from_wire.size() << std::endl;
                std::cout << "Number of xyz distance from gas: " << xyz_distance_from_gas.size() << std::endl;
                std::cout << "Number of xyz distance from source gap: " << xyz_distance_from_source_gap.size() << std::endl;
            }
        }


		// ELEC CROSSING THE SOURCE FOILE i.e electron cluster with alpha like cluster within < 10 µs delta t and space corelation (there are in different sides of the source foil) 
		side_elec = -1; 
		side_UND = -1;
		tracks_elec_crossing = false;
		nb_of_elec_crossing = 0;
		vector<TrackMatch> potential_matches;
		elec_used.assign(nb_of_elec_candidates, false);
		und_used.assign(nb_of_UND_candidates, false);
		if(tracks_with_associated_calo && tracks_without_associated_calo){

			for(int j = 0; j < nb_of_elec_candidates; j++) {
				// Vérifier le côté de l'électron
				int side_elec = (cell_num_per_elec_cluster.at(j).front() < 1017) ? 0 : 1;

				if(nb_fit_solution_per_elec_cluster.at(j) >= 1 && nb_of_OM_per_elec_cluster.at(j) == 1) {
					for(int l = 0; l < nb_fit_solution_per_elec_cluster.at(j); l++) {// Loop in elec fit solutions 
						// Lopp
						for(int m = 0; m < nb_of_UND_candidates; m++) { // Lopp in UND candidates
							int side_UND = (cell_num_per_UND_cluster.at(m).front() < 1017) ? 0 : 1;

							// Vérifier que les traces sont de côtés opposés
							if((side_elec == 1 && side_UND == 0) || (side_elec == 0 && side_UND == 1)) {
								// Vérifier la proximité temporelle
								double time_diff = abs((min_time_per_UND_cluster.at(m)) - (OM_timestamp_per_elec_cluster.at(j).at(0)))  ; // wtf les times... c'est pas en us
								//std::cout<<"================ EVENT : "<<event_number<<"================ "<<ptd_event<<std::endl;
								//std::cout<<"min_time_per_UND_cluster.at(m) = "<<std::setprecision(25)<<min_time_per_UND_cluster.at(m)<<std::endl;
								//std::cout<<"OM_timestamp_per_elec_cluster.at(j).front() = "<<std::setprecision(25)<<OM_timestamp_per_elec_cluster.at(j).front()<<std::endl;
								//std::cout<<"time_diff = "<<std::setprecision(25)<<time_diff<<std::endl;

								if(time_diff <= 10000000E-6) { // To be justified

									//std::cout<<" OUI !"<<std::endl;
									for(int n = 0; n < nb_fit_solution_per_UND_cluster.at(m); n++) { // Lopp in UND fit solutions
										double spatial_distance;
										
										// Si les deux traces ont un vertex sur le plan source
										if(vertex_is_on_reference_source_plane_per_elec_cluster.at(j).at(l) == 1 &&
										   vertex_is_on_reference_source_plane_per_UND_cluster.at(m).at(n) == 1) {
											//std::cout<<"==> 	vertex on reference source plance used ! "<<std::endl;
											spatial_distance = calculateDistance(
												x_is_on_reference_source_plane.at(j).at(l),
												y_is_on_reference_source_plane.at(j).at(l),
												z_is_on_reference_source_plane.at(j).at(l),
												x_is_on_reference_source_plane_per_UND_cluster.at(m).at(n),
												y_is_on_reference_source_plane_per_UND_cluster.at(m).at(n),
												z_is_on_reference_source_plane_per_UND_cluster.at(m).at(n)
											);

											// Si la distance est raisonnable (à ajuster selon vos besoins)
											if(spatial_distance < 5000000) { // en mm 
												TrackMatch match;
												match.elec_idx = j;
												match.und_idx = m;
												match.elec_fit_idx = l;
												match.und_fit_idx = n;
												match.spatial_distance = spatial_distance;
												match.time_difference = time_diff;
												if(side_UND ==0){
													match.closest_point = "start";
												}
												else{
													match.closest_point = "end";
												}

												potential_matches.push_back(match);
											}
										}
										// Sinon utiliser les derniers points
										else{
											//std::cout<<"==> 	last points used ! "<<std::endl;

											double spatial_distance_start, spatial_distance_end, spatial_distance;

											if (cell_num_per_elec_cluster.at(j).front() < 1017) {
												// Compare start of elec with start and end of UND
												spatial_distance_start = calculateDistance(
													x_start_per_elec_cluster.at(j).at(l),
													y_start_per_elec_cluster.at(j).at(l),
													z_start_per_elec_cluster.at(j).at(l),
													x_start_per_UND_cluster.at(m).at(n),
													y_start_per_UND_cluster.at(m).at(n),
													z_start_per_UND_cluster.at(m).at(n)
												);

												spatial_distance_end = calculateDistance(
													x_start_per_elec_cluster.at(j).at(l),
													y_start_per_elec_cluster.at(j).at(l),
													z_start_per_elec_cluster.at(j).at(l),
													x_end_per_UND_cluster.at(m).at(n),
													y_end_per_UND_cluster.at(m).at(n),
													z_end_per_UND_cluster.at(m).at(n)
												);
											} 
											else if(1==0) {
												// Compare end of elec with start and end of UND
												spatial_distance_start = calculateDistance(
													x_end_per_elec_cluster.at(j).at(l),
													y_end_per_elec_cluster.at(j).at(l),
													z_end_per_elec_cluster.at(j).at(l),
													x_start_per_UND_cluster.at(m).at(n),
													y_start_per_UND_cluster.at(m).at(n),
													z_start_per_UND_cluster.at(m).at(n)
												);

												spatial_distance_end = calculateDistance(
													x_end_per_elec_cluster.at(j).at(l),
													y_end_per_elec_cluster.at(j).at(l),
													z_end_per_elec_cluster.at(j).at(l),
													x_end_per_UND_cluster.at(m).at(n),
													y_end_per_UND_cluster.at(m).at(n),
													z_end_per_UND_cluster.at(m).at(n)
												);
											}

											// Determine the closest point (start or end)
											spatial_distance = std::min(spatial_distance_start, spatial_distance_end);

											if (spatial_distance <= 50000) { // Adjust threshold as needed
												TrackMatch match;
												match.elec_idx = j;
												match.und_idx = m;
												match.elec_fit_idx = l;
												match.und_fit_idx = n;
												match.spatial_distance = spatial_distance;
												match.time_difference = time_diff;

												//Store additional information about the closest point
												if (spatial_distance == spatial_distance_start) {
													match.closest_point = "start";
												} else {
													match.closest_point = "end";
												}

												potential_matches.push_back(match);
											}
										}
									}
								}
							}
						}
					}
				}
			}
			// Après avoir rempli potential_matches

			// Trie les matches par distance spatiale croissante
			std::sort(potential_matches.begin(), potential_matches.end(),
			[](const TrackMatch& a, const TrackMatch& b) {
				return a.spatial_distance < b.spatial_distance;
			});

			// Vecteurs pour savoir si un elec ou un UND a déjà été associé

			//std::vector<bool> elec_used(nb_of_elec_candidates, false); // ??
			//std::vector<bool> und_used(nb_of_UND_candidates, false); // ?? ICI comme ça ? 
			elec_used.assign(nb_of_elec_candidates, false); // WARNING .assign can be time consuming (more than .resize() )
			und_used.assign(nb_of_UND_candidates, false); // WARNING .assign can be time consuming (more than .resize() )


			std::vector<TrackMatch> best_matches;

			for(const auto& match : potential_matches) {
				if(!elec_used[match.elec_idx] && !und_used[match.und_idx]) {
					best_matches.push_back(match);
					elec_used[match.elec_idx] = true;
					und_used[match.und_idx] = true;
				}
			}
			std::cout<<"SIZE elec_used = "<<elec_used.size()<<std::endl;
			for(int i = 0; i <elec_used.size(); i++){
				std:cout<<"elec_used["<<i<<"] = "<<elec_used[i]<<std::endl;

			}
			// Affiche les associations retenues
			//std::cout << "Best unique matches (one per electron cluster):" << std::endl;

			
			//std::cout<<"		 !! --> NB of best matches = "<<best_matches.size()<<std::endl;
			if(best_matches.size() >= 1) {
				std::cout<<"		 ==================================================================== LAAAAAAAAAAAAAAA =============================================== "<<std::endl;
				tracks_elec_crossing = true;
			}
			nb_of_elec_crossing = best_matches.size();

			for(size_t idx = 0; idx < best_matches.size(); ++idx){
				const TrackMatch& match = best_matches[idx];
				//std::cout << "Best Match " << idx
				//<< " | elec_idx: " << match.elec_idx
				//<< " | und_idx: " << match.und_idx
				//<< " | elec_fit_idx: " << match.elec_fit_idx
				//<< " | und_fit_idx: " << match.und_fit_idx
				//<< " | spatial_distance: " << match.spatial_distance
				//<< " | time_difference: " << match.time_difference
				//<< std::endl;
				
				fit_elec_index_for_each_elec_crossing.push_back(match.elec_idx);

				distance_elec_UND_per_elec_crossing.push_back({match.spatial_distance});

				// We need Tracks information as Cluster ID, idx electron and fit

				cell_num_per_elec_crossing.push_back(
					merge_vectors(cell_num_per_elec_cluster.at(match.elec_idx),
								  cell_num_per_UND_cluster.at(match.und_idx)));
			
				anode_time_per_elec_crossing.push_back(
					merge_vectors(anode_time_per_elec_cluster.at(match.elec_idx),
								  anode_time_per_UND_cluster.at(match.und_idx)));
			
				top_cathode_per_elec_crossing.push_back(
					merge_vectors(top_cathode_per_elec_cluster.at(match.elec_idx),
								  top_cathodes_per_UND_cluster.at(match.und_idx)));
			
				bottom_cathode_per_elec_crossing.push_back(
					merge_vectors(bottom_cathode_per_elec_cluster.at(match.elec_idx),
								  bottom_cathodes_per_UND_cluster.at(match.und_idx)));
			
				z_of_cells_per_elec_crossing.push_back(
					merge_vectors(z_of_cells_per_elec_cluster.at(match.elec_idx),
								  z_of_cell_per_UND_cluster.at(match.und_idx)));
			
				sigma_z_of_cells_per_elec_crossing.push_back(
					merge_vectors(sigma_z_of_cells_per_elec_cluster.at(match.elec_idx),
								  sigma_z_of_cells_per_UND_cluster.at(match.und_idx)));
			
				r_of_cells_per_elec_crossing.push_back(
					merge_vectors(r_of_cells_per_elec_cluster.at(match.elec_idx),
								  r_of_cells_per_UND_cluster.at(match.und_idx)));
			
				sigma_r_of_cells_per_elec_crossing.push_back(
					merge_vectors(sigma_r_of_cells_per_elec_cluster.at(match.elec_idx),
								  sigma_r_of_cells_per_UND_cluster.at(match.und_idx)));

				
				if(match.closest_point == "start") {
					//std::cout<<"		==> 	start point used ! "<<std::endl;
					x_start_per_elec_crossing.push_back({x_end_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx)});
					y_start_per_elec_crossing.push_back({y_end_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx)});
					z_start_per_elec_crossing.push_back({z_end_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx)});
					//std::cout<<"x_end_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx) = "<<x_end_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx)<<std::endl;
					//std::cout<<"y_end_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx) = "<<y_end_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx)<<std::endl;
					//std::cout<<"z_end_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx) = "<<z_end_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx)<<std::endl;

				} 
				if(match.closest_point == "end") {
					//std::cout<<"		==> 	end point used ! "<<std::endl;
					x_start_per_elec_crossing.push_back({x_start_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx)});
					y_start_per_elec_crossing.push_back({y_start_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx)});
					z_start_per_elec_crossing.push_back({z_start_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx)});
					//std::cout<<"x_start_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx) = "<<x_start_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx)<<std::endl;
					//std::cout<<"y_start_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx) = "<<y_start_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx)<<std::endl;
					//std::cout<<"z_start_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx) = "<<z_start_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx)<<std::endl;

				}
				if(cell_num_per_elec_cluster.at(match.elec_idx).front() < 1017 && cell_num_per_UND_cluster.at(match.und_idx).front() >= 1017){
					//std::cout<<"		==> 	elec side 0 ! ==> END Used of elec"<<std::endl;
					x_end_per_elec_crossing.push_back({x_end_per_elec_cluster.at(match.elec_idx).at(match.elec_fit_idx)});
					y_end_per_elec_crossing.push_back({y_end_per_elec_cluster.at(match.elec_idx).at(match.elec_fit_idx)});
					z_end_per_elec_crossing.push_back({z_end_per_elec_cluster.at(match.elec_idx).at(match.elec_fit_idx)});
					
				}
				if(cell_num_per_elec_cluster.at(match.elec_idx).front() >= 1017 && cell_num_per_UND_cluster.at(match.und_idx).front() < 1017){
					//std::cout<<"		==> 	elec side 1 ! ==> START Used for elec end "<<std::endl;
					x_end_per_elec_crossing.push_back({x_start_per_elec_cluster.at(match.elec_idx).at(match.elec_fit_idx)});
					y_end_per_elec_crossing.push_back({y_start_per_elec_cluster.at(match.elec_idx).at(match.elec_fit_idx)});
					z_end_per_elec_crossing.push_back({z_start_per_elec_cluster.at(match.elec_idx).at(match.elec_fit_idx)});
				}		
				
				OM_num_per_elec_crossing.push_back({OM_num_per_elec_cluster.at(match.elec_idx).at(0)});
				E_OM_per_elec_crossing.push_back({E_OM_per_elec_cluster.at(match.elec_idx).at(0)});
				OM_timestamp_per_elec_crossing.push_back({OM_timestamp_per_elec_cluster.at(match.elec_idx).at(0)});
				OM_charge_per_elec_crossing.push_back({OM_charge_per_elec_cluster.at(match.elec_idx).at(0)});
				OM_amplitude_per_elec_crossing.push_back({OM_amplitude_per_elec_cluster.at(match.elec_idx).at(0)});
				OM_LT_only_per_elec_crossing.push_back({OM_LT_only_per_elec_cluster.at(match.elec_idx).at(0)});
				OM_HT_per_elec_crossing.push_back({OM_HT_per_elec_cluster.at(match.elec_idx).at(0)});

				ID_cluster_per_elec_crossing.push_back(ID_clsuter_per_elec_cluster.at(match.elec_idx));					// NEED TO STORE THIS IN THE TREE ! WRITE DONE WHEN IT'LL BE DONE
				ID_cluster_UND_per_elec_crossing.push_back(ID_clsuter_UND.at(match.und_idx));					// NEED TO STORE THIS IN THE TREE ! WRITE DONE WHEN IT'LL BE DONE
				
				double elec_track_length = calculateDistance(
					x_start_per_elec_cluster.at(match.elec_idx).at(match.elec_fit_idx),
					y_start_per_elec_cluster.at(match.elec_idx).at(match.elec_fit_idx),
					z_start_per_elec_cluster.at(match.elec_idx).at(match.elec_fit_idx),
					x_end_per_elec_cluster.at(match.elec_idx).at(match.elec_fit_idx),
					y_end_per_elec_cluster.at(match.elec_idx).at(match.elec_fit_idx),
					z_end_per_elec_cluster.at(match.elec_idx).at(match.elec_fit_idx)
				);

				double und_track_length = calculateDistance(
					x_start_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx),
					y_start_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx),
					z_start_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx),
					x_end_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx),
					y_end_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx),
					z_end_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx)
				);

				// Should add the delta distance between extremities of the two tracks... not adding xyz distance ?? 
				double total_track_length = elec_track_length + und_track_length + xyz_distance_from_reference_source_plane.at(match.elec_idx).at(match.elec_fit_idx) + xyz_distance_from_reference_source_plane_per_UND_cluster.at(match.und_idx).at(match.und_fit_idx);



				// ONly one element
				lenght_per_elec_crossing.push_back({total_track_length});
				
				vector<double> temp_delta_t_cells_of_UND_per_elec_crossing;
				for(int k = 0; k < nb_cell_per_UND_cluster.at(match.und_idx); k++){
					//std::cout<<"delta_t = "<<abs((anode_time_per_UND_cluster.at(match.und_idx).at(k) )*1E-9- (OM_timestamp_per_elec_cluster.at(match.elec_idx).at(0) *1E-9 ) )<<std::endl;
					//std::cout<<"OM_timestamp_per_elec_cluster.at(match.elec_idx).size() = "<<OM_timestamp_per_elec_cluster.at(match.elec_idx).size()<<std::endl;
					temp_delta_t_cells_of_UND_per_elec_crossing.push_back(abs((anode_time_per_UND_cluster.at(match.und_idx).at(k)  )- (OM_timestamp_per_elec_cluster.at(match.elec_idx).at(0)  ) )); //us

				}
				delta_t_cells_of_UND_per_elec_crossing.push_back(temp_delta_t_cells_of_UND_per_elec_crossing);
				vector<double> temp_delta_t_cells_of_elec_per_elec_crossing;
				for(int k = 0; k < nb_cell_per_elec_cluster.at(match.elec_idx); k++){
					temp_delta_t_cells_of_elec_per_elec_crossing.push_back(abs((anode_time_per_elec_cluster.at(match.elec_idx).at(k)  )- (OM_timestamp_per_elec_cluster.at(match.elec_idx).at(0)  )));
				}

				delta_t_cells_of_elec_per_elec_crossing.push_back(temp_delta_t_cells_of_elec_per_elec_crossing);
			}
		}
		
		for(int k = 0; k < cell_num_per_elec_crossing.size(); k++){
			nb_of_cell_per_elec_crossing.push_back(cell_num_per_elec_crossing.at(k).size());
		
		}	
			
		// Here we gonna check if electrons / elec crossing and UNDs candidate can't be  associated
		int nb_elec_used = 0;
		int nb_UND_used = 0;
		for(int j = 0 ; j < nb_of_elec_candidates; j ++){
			if(elec_used[j] == true){
				nb_elec_used ++;
				continue;
			}
		}

		
		for(int k = 0; k < nb_of_UND_candidates; k++){
			if(und_used[k] == true){
				nb_UND_used ++;
				continue;
			}
		}
		
		nb_tot_elec = nb_of_elec_candidates - nb_elec_used; 
		nb_UND_tot = nb_of_UND_candidates - nb_UND_used;
		
		
		if(DEBUG == true){
			std::cout<<"\n======================== ELEC CROSSING DEBUG ===================== \n"<<std::endl;
			
			std::cout<<"Nb of elec crossing: "<<nb_of_elec_crossing<<std::endl;

			for(int k = 0; k < nb_of_elec_crossing; k++){
				std::cout<<"	==> ELECTRON CROSSING "<<k<<std::endl;
				std::cout<<"nb_of_cell_per_elec_crossing = "<<nb_of_cell_per_elec_crossing.at(k)<<std::endl;	
				std::cout<<"OM_num_per_elec_crossing.at(k).size() = "<<OM_num_per_elec_crossing.at(k).size()<<std::endl;
				
				std::cout<<"	==> OM_num_per_elec_crossing.at(k).front() = "<<OM_num_per_elec_crossing.at(k).front() <<std::endl;
				
				std::cout<<" 	==> E_OM_per_elec_crossing.at(k).fron() = "<< E_OM_per_elec_crossing.at(k).front() <<std::endl;

				for(int l = 0; l < cell_num_per_elec_crossing.at(k).size(); l++){
					
					std::cout<<"	==> NUM CELL = "<<cell_num_per_elec_crossing.at(k).at(l)<<std::endl;
					std::cout<<"			R0 = "<<anode_time_per_elec_crossing.at(k).at(l) <<std::endl;
					std::cout<<"			R5 = "<<top_cathode_per_elec_crossing.at(k).at(l) <<std::endl;
					std::cout<<"			R6 = "<<bottom_cathode_per_elec_crossing.at(k).at(l) <<std::endl;
					std::cout<<"			z = "<<z_of_cells_per_elec_crossing.at(k).at(l) <<std::endl;
					std::cout<<"\n"<<std::endl;

				}
				std::cout<<"\n=============3D information=============\n"<<std::endl;

				std::cout<<"x_start_per_elec_crossing.at(k).size() = "<<x_start_per_elec_crossing.at(k).size()<<std::endl;
				std::cout<<"y_start_per_elec_crossing.at(k).size() = "<<y_start_per_elec_crossing.at(k).size()<<std::endl;
				std::cout<<"z_start_per_elec_crossing.at(k).size() = "<<z_start_per_elec_crossing.at(k).size()<<std::endl;

				for(int l = 0; l < x_start_per_elec_crossing.at(k).size(); l++){
					std::cout<<"\n l = "<<l<<"\n"<<std::endl;
					std::cout<<"	==> x_start_per_elec_crossing.at(k).at(l) = "<<x_start_per_elec_crossing.at(k).at(l)<<std::endl;
					std::cout<<"	==> y_start_per_elec_crossing.at(k).at(l) = "<<y_start_per_elec_crossing.at(k).at(l)<<std::endl;
					std::cout<<"	==> z_start_per_elec_crossing.at(k).at(l) = "<<z_start_per_elec_crossing.at(k).at(l)<<std::endl;

					std::cout<<"\nLAST\n"<<std::endl;
					std::cout<<"	==> x_end_per_elec_crossing.at(k).at(l) = "<<x_end_per_elec_crossing.at(k).at(l)<<std::endl;
					std::cout<<"	==> y_end_per_elec_crossing.at(k).at(l) = "<<y_end_per_elec_crossing.at(k).at(l)<<std::endl;
					std::cout<<"	==> z_end_per_elec_crossing.at(k).at(l) = "<<z_end_per_elec_crossing.at(k).at(l)<<std::endl;
				}
			}

		}

		if(DEBUG == true){
			std::cout<<"nb_of_elec_candidates = "<<nb_of_elec_candidates<<std::endl;
			std::cout<<"nb_of_elec_crossing = "<<nb_of_elec_crossing<<std::endl;
			std::cout<<"nb_of_UND_candidates = "<<nb_of_UND_candidates<<std::endl;
			
			for(int i =0; i < nb_of_elec_candidates; i++){

				if(elec_used[i] == true){
					std::cout<<elec_used[i]<<std::endl;
					std::cout<<" i = "<<i<<std::endl;
					std::cout<<"Elec candidate with ID = "<<ID_clsuter_per_elec_cluster[i]<<" is used !"<<std::endl;
				}
			}

			for(int i = 0; i < nb_of_UND_candidates; i++){
				std::cout<<und_used[i]<<std::endl;
				if(und_used[i] ==true ){

					std::cout<<"UND candidate with ID = "<<ID_clsuter_UND[i]<<" is used !"<<std::endl;
				}
			}
		}
		// Gona try to replace electron cluster with elec crossing information.
/* 
		if(nb_of_elec_crossing>0){
			for(int i = 0; i < nb_of_elec_candidates; i++){
				if(elec_used[i] == true){
					//std::cout<<"elec_used["<<i<<"] = "<<elec_used[i]<<std::endl;
					cell_num_per_elec_cluster[i] = cell_num_per_elec_crossing[i];
					anode_time_per_elec_cluster[i] = anode_time_per_elec_crossing[i];
					top_cathode_per_elec_cluster[i] = top_cathode_per_elec_crossing[i];
					bottom_cathode_per_elec_cluster[i] = bottom_cathode_per_elec_crossing[i];
					z_of_cells_per_elec_cluster[i] = z_of_cells_per_elec_crossing[i];
					sigma_z_of_cells_per_elec_cluster[i] = sigma_z_of_cells_per_elec_crossing[i];
					r_of_cells_per_elec_cluster[i] = r_of_cells_per_elec_crossing[i];
					sigma_r_of_cells_per_elec_cluster[i] = sigma_r_of_cells_per_elec_crossing[i];
					x_start_per_elec_cluster[i] = x_start_per_elec_crossing[i];
					y_start_per_elec_cluster[i] = y_start_per_elec_crossing[i];
					z_start_per_elec_cluster[i] = z_start_per_elec_crossing[i];
					x_end_per_elec_cluster[i] = x_end_per_elec_crossing[i];
					y_end_per_elec_cluster[i] = y_end_per_elec_crossing[i];
					z_end_per_elec_cluster[i] = z_end_per_elec_crossing[i];
					OM_num_per_elec_cluster[i] = OM_num_per_elec_crossing[i];
					E_OM_per_elec_cluster[i] = E_OM_per_elec_crossing[i];
					OM_timestamp_per_elec_cluster[i] = OM_timestamp_per_elec_crossing[i];
					OM_charge_per_elec_cluster[i] = OM_charge_per_elec_crossing[i];
					OM_amplitude_per_elec_cluster[i] = OM_amplitude_per_elec_crossing[i];
					OM_LT_only_per_elec_cluster[i] = OM_LT_only_per_elec_crossing[i];
					OM_HT_per_elec_cluster[i] = OM_HT_per_elec_crossing[i];

					// Remove duplicate
                		
						
  						
  					nb_of_OM_per_elec_cluster.erase(nb_of_OM_per_elec_cluster.begin() + j);
					elec_cluster_is_delayed.erase(elec_cluster_is_delayed.begin() + j);
					elec_clsuter_is_prompt.erase(elec_clsuter_is_prompt.begin() + j);
						
						
					
					
					
					
					
					
						
					x_is_on_reference_source_plane[i].push_back(x_is_on_reference_source_plane[j][0]);
					y_is_on_reference_source_plane[i].push_back(y_is_on_reference_source_plane[j][0]);
					z_is_on_reference_source_plane[i].push_back(z_is_on_reference_source_plane[j][0]);
						
				
					x_is_on_source_foil[i].push_back(x_is_on_source_foil[j][0]);
						
					y_is_on_source_foil[i].push_back(y_is_on_source_foil[j][0]);
					z_is_on_source_foil[i].push_back(z_is_on_source_foil[j][0]);
					x_is_on_main_calorimeter[i].push_back(x_is_on_main_calorimeter[j][0]);
					y_is_on_main_calorimeter[i].push_back(y_is_on_main_calorimeter[j][0]);
					z_is_on_main_calorimeter[i].push_back(z_is_on_main_calorimeter[j][0]);
					x_is_on_x_calorimeter[i].push_back(x_is_on_x_calorimeter[j][0]);
						
					y_is_on_x_calorimeter[i].push_back(y_is_on_x_calorimeter[j][0]);
					z_is_on_x_calorimeter[i].push_back(z_is_on_x_calorimeter[j][0]);
					x_is_on_gamma_veto[i].push_back(x_is_on_gamma_veto[j][0]);
					y_is_on_gamma_veto[i].push_back(y_is_on_gamma_veto[j][0]);
					z_is_on_gamma_veto[i].push_back(z_is_on_gamma_veto[j][0]);
						
					x_is_on_wire[i].push_back(x_is_on_wire[j][0]);
					y_is_on_wire[i].push_back(y_is_on_wire[j][0]);
					z_is_on_wire[i].push_back(z_is_on_wire[j][0]);
						
					x_is_in_gas[i].push_back(x_is_in_gas[j][0]);
					y_is_in_gas[i].push_back(y_is_in_gas[j][0]);
					z_is_in_gas[i].push_back(z_is_in_gas[j][0]);
					x_is_on_source_gap[i].push_back(x_is_on_source_gap[j][0]);
					y_is_on_source_gap[i].push_back(y_is_on_source_gap[j][0]);
					z_is_on_source_gap[i].push_back(z_is_on_source_gap[j][0]);

						


					xy_distance_from_reference_source_plane[i].push_back(xy_distance_from_reference_source_plane[j][0]);
					xy_distance_from_source_foil[i].push_back(xy_distance_from_source_foil[j][0]);
					xy_distance_from_main_calorimeter[i].push_back(xy_distance_from_main_calorimeter[j][0]);
					xy_distance_from_x_calorimeter[i].push_back(xy_distance_from_x_calorimeter[j][0]);
					xy_distance_from_gamma_veto[i].push_back(xy_distance_from_gamma_veto[j][0]);
					xy_distance_from_wire[i].push_back(xy_distance_from_wire[j][0]);
					xy_distance_from_gas[i].push_back(xy_distance_from_gas[j][0]);
					xy_distance_from_source_gap[i].push_back(xy_distance_from_source_gap[j][0]);

					xyz_distance_from_reference_source_plane[i].push_back(xyz_distance_from_reference_source_plane[j][0]);
					xyz_distance_from_source_foil[i].push_back(xyz_distance_from_source_foil[j][0]);
					xyz_distance_from_main_calorimeter[i].push_back(xyz_distance_from_main_calorimeter[j][0]);
					xyz_distance_from_x_calorimeter[i].push_back(xyz_distance_from_x_calorimeter[j][0]);
					xyz_distance_from_gamma_veto[i].push_back(xyz_distance_from_gamma_veto[j][0]);
					xyz_distance_from_wire[i].push_back(xyz_distance_from_wire[j][0]);
					xyz_distance_from_gas[i].push_back(xyz_distance_from_gas[j][0]);
					xyz_distance_from_source_gap[i].push_back(xyz_distance_from_source_gap[j][0]);


					vertex_is_on_reference_source_plane_per_elec_cluster[i].push_back(vertex_is_on_reference_source_plane_per_elec_cluster[j][0]);
					vertex_is_on_source_foil_per_elec_cluster[i].push_back(vertex_is_on_source_foil_per_elec_cluster[j][0]);
					vertex_is_on_main_calorimeter_per_elec_cluster[i].push_back(vertex_is_on_main_calorimeter_per_elec_cluster[j][0]);
					vertex_is_on_x_calorimeter_per_elec_cluster[i].push_back(vertex_is_on_x_calorimeter_per_elec_cluster[j][0]);
					vertex_is_on_gamma_veto_per_elec_cluster[i].push_back(vertex_is_on_gamma_veto_per_elec_cluster[j][0]);
					vertex_is_on_wire_per_elec_cluster[i].push_back(vertex_is_on_wire_per_elec_cluster[j][0]);
					vertex_is_in_gas_per_elec_cluster[i].push_back(vertex_is_in_gas_per_elec_cluster[j][0]);
					vertex_is_on_source_gap_per_elec_cluster[i].push_back(vertex_is_on_source_gap_per_elec_cluster[j][0]);


						
					

				}
			}
		} 
		*/
		// KEEP UND INFORMATIONS OF ELEC CROSSING ==> GONA USE TAG FROM UND USED ! :) 

		
		

		if((evt_isolated_calo == true || tracks_with_associated_calo==true || tracks_without_associated_calo == true) && saving_file == true){  
			tree->Fill();
      	}
      	event_number++;
      	if(event_number%100000 ==0 && saving_file==true){
			std::cout<<event_number<<" proceed ! "<<std::endl;
      	}
    }//End if ptd_details 

  	return dpp::base_module::PROCESS_SUCCESS;
}

