import requests
import pandas as pd
import pickle
import geopandas as gpd
from shapely.ops import nearest_points
from shapely.geometry import LineString, Point
import scipy.stats as st
from clustergram import Clustergram
import urbangrammar_graphics as ugg
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import scale, MinMaxScaler
import numpy as np


# IMPORTING DATA
with open('data/team_id_dict.pickle', 'rb') as handle:
    team_ids = pickle.load(handle)

# FUNCTIONS
##### API Calls Noteboook #####
def make_bare_api_call(additional_text):
    assert type(additional_text) == str
    url = 'https://statsapi.web.nhl.com/api/v1/'
    response = requests.request("GET", url + additional_text)
    return response.json()

def find_player_ids(team_roster_json, team_id):
    assert type(team_roster_json) == dict
    assert type(team_id) == int
    player_list = []
    for x in range(len(team_roster_json['roster'])):
        player_dict = team_roster_json['roster'][x]['person']
        player_dict['position'] = team_roster_json['roster'][x]['position']['abbreviation']
        player_list.append(player_dict)
    return player_list

def team_ids_to_players_df(team_id):
    assert type(team_id) == int
    teams_json = make_bare_api_call(f"/teams/{team_id}/roster")
    player_list = find_player_ids(teams_json, team_id)
    player_list_df = pd.DataFrame.from_dict(player_list)
    player_list_df['team_id'] = team_id
    return player_list_df

def get_all_player_ids_into_df(team_ids=team_ids):
    team_df_list = []
    for item in team_ids.values():
        team_df = team_ids_to_players_df(item)
        team_df_list.append(team_df)
    # make the whole team_df into a single df
    all_players_df = pd.concat(team_df_list)
    return all_players_df.reset_index(drop=True)

def add_player_bithplace_to_df(df):
    part_of_api_list = [df['link'][x][8:] for x in range(df.shape[0])]
    player_info_combined_list = []
    for partial_API in part_of_api_list:
        player_info_json = make_bare_api_call(partial_API)
        province_info = player_info_json['people'][0].get('birthStateProvince')
        player_info = [player_info_json['people'][0]['birthCity'],
                       province_info,
                       player_info_json['people'][0]['birthCountry'],
                       player_info_json['people'][0]['nationality']]
        player_info_combined_list.append(player_info)
    # unzip the player_info
    unzipped_player_info_combined_list = list(zip(*player_info_combined_list))
    # add to dataframe
    df['birthCity'] = unzipped_player_info_combined_list[0]
    df['birthStateProvince'] = unzipped_player_info_combined_list[1]
    df['birthCountry'] = unzipped_player_info_combined_list[2]
    df['nationality'] = unzipped_player_info_combined_list[3]
    return df

##### Finding City Locations #####
def find_city_coordinates(player_df, city_df):
    coordinates = []
    for idx in range(player_df.shape[0]):
        player_city_prov = player_df['city_province'][idx]
        coords = None
        if city_df['city_province'].isin([player_city_prov]).any() == True:
            city_idx = city_df.loc[city_df['city_province'] == player_city_prov].index[0]
            coords = str(city_df['lat'][city_idx]) + ',' + str(city_df['lng'][city_idx])
        coordinates.append(coords)
    return coordinates

def turn_coords_to_correct_format(data):
    assert type(data) == str
    new_str = ""
    acceptable_item_list = [",", ".", "0", "1", "2", "3", "4",
                           "5", "6", "7", "8", "9"]
    for item in data:
        if item in acceptable_item_list:
            new_str += item
    final_str = new_str[0:8] + '-' + new_str[8:]
    return final_str

##### Calculating Nearest Neighbor #####
# Note: most of these functions come from the following repo: 
# https://github.com/shakasom/NearestNeighbour-Analysis/blob/master/NNA.ipynb
def create_gdf(df, x="Lat", y="Lng"):
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x], df[y]), crs={"init":"epsg:4326"})

def calculate_nearest(row, destination, val, col="geometry"):
    dest_unary = destination["geometry"].unary_union
    nearest_geom = nearest_points(row[col], dest_unary)
    match_geom = destination.loc[destination.geometry == nearest_geom[1]]
    match_value = match_geom[val].to_numpy()[0]
    return match_value

##### Running Stats on Players per capita distributions #####
#taken from https://stackoverflow.com/questions/37487830/how-to-find-probability-distribution-and-parameters-for-real-data-python-3
def get_best_distribution(data):
#     dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    # this is from https://towardsdatascience.com/identify-your-datas-distribution-d76062fc0802
    dist_names = ['weibull_min','norm','weibull_max','beta',
              'invgauss','uniform','gamma','expon',   
              'lognorm','pearson3', 'triang']
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]

##### Optimal number of cluster #####
# both of these were made using help from the clustergram page https://clustergram.readthedocs.io/en/latest/
def fit_plot_clustergram(scaled_data, end_range=10):
    cgram = Clustergram(range(1, end_range), n_init=1000, verbose=False)
    cgram.fit(scaled_data)
    ax = cgram.plot(
    figsize=(10, 8),
    line_style=dict(color=ugg.COLORS[1]),
    cluster_style={"color": ugg.COLORS[2]}
    )
    ax.yaxis.grid(False)
    sns.despine(offset=10)
    return cgram

def visualize_num_of_clusters(clustergram):  
    fig, axs = plt.subplots(2, figsize=(10, 10), sharex=True)
    # score of 1 is best with the silhouette score
    clustergram.silhouette_score().plot(
        color=ugg.COLORS[1],
        ax=axs[0],
    )
    # the higher the value, the better
    clustergram.calinski_harabasz_score().plot(
        color=ugg.COLORS[1],
        ax=axs[1]
    )
    sns.despine(offset=10)
    
def evaluate_num_of_clusters(clustergram):
    # create silhouette_score 
    min_max_scaler = MinMaxScaler()
    silhouette_score_data = np.array(clustergram.silhouette_score()).reshape(-1, 1)
    min_max_scaler.fit(silhouette_score_data)
    scaled_sil_data = min_max_scaler.transform(silhouette_score_data)
    # get calinski_harabasz_score data 
    # min_max_scaler_c = MinMaxScaler()
    calinski_score_data = np.array(clustergram.calinski_harabasz_score()).reshape(-1, 1)
    min_max_scaler.fit(calinski_score_data)
    scaled_cal_data = min_max_scaler.transform(calinski_score_data)
    cluster_stat_combined = scaled_sil_data + scaled_cal_data
    # index starts at 2
    best_index = np.where(cluster_stat_combined == cluster_stat_combined.max())
    # return the list of values produced and the best index
    return cluster_stat_combined, best_index[0][0] + 2
    
def make_dbscan_cluster(esp_param, min_samples_param, data):
    dbscan_model = DBSCAN(eps=esp_param, min_samples=min_samples_param)
    dbscan_yhat = dbscan_model.fit_predict(data)
    return dbscan_yhat

def determine_if_noise(dbscan_predictions):
    # determining if the player's birthplace was placed into a cluster, or was considered noise
    noisey_data = np.where(dbscan_predictions==-1, 0, 1)
    return noisey_data

def run_cluster_determine_noise(esp_param, min_samples_param, data):
    dbscan_predictions = make_dbscan_cluster(esp_param, min_samples_param, data)
    noise_determination = determine_if_noise(dbscan_predictions)
    return noise_determination
    
##### Plotting the clusters #####
def add_kmeans_centroids_to_gdf(clustered_gdf, fitted_kmeans):
    num_clusters = len(fitted_kmeans.cluster_centers_)
    if 'fullName' in clustered_gdf.columns:
        centroid_gdf = gpd.GeoDataFrame(data=[f"Cluster{x}" for x in range(0, num_clusters)], columns = ['fullName'])
    elif 'Team_name' in clustered_gdf.columns:
        centroid_gdf = gpd.GeoDataFrame(data=[f"Cluster{x}" for x in range(0, num_clusters)], columns = ['Team_name'])
    # making the centroids into points
    centroid_gdf['geometry'] = [Point(coord[1], coord[0]) for coord in fitted_kmeans.cluster_centers_]
    if 'player_cluster' in clustered_gdf.columns:
        centroid_gdf['player_cluster'] = [x for x in range(num_clusters, (num_clusters*2))]
    elif 'arena_cluster' in clustered_gdf.columns:
        centroid_gdf['arena_cluster'] = [x for x in range(num_clusters, (num_clusters*2))]
    clustered_gdf_w_centroids = pd.concat([clustered_gdf, centroid_gdf])
    return clustered_gdf_w_centroids

def plot_clusters(us_can_gdf, clustered_gdf, col_name_to_plot: str, include_legend=False):
    ax = us_can_gdf.plot(figsize = (12, 12), facecolor='white', edgecolor='grey')
    clustered_gdf.plot(column=col_name_to_plot, cmap='tab20b', ax=ax, figsize = (12, 12), legend=include_legend)
    return ax

##### Finding City Name for Each Cluster #####
def assign_clusters_names(dataframe, city_name_list: list):
    cluster_to_name_dict = {str(x): None for x in dataframe['player_cluster_dbscan'].unique()}
    counter = 0
    while counter < len(cluster_to_name_dict):
        for idx in range(len(dataframe)):
            if dataframe['city_province'].iloc[idx] in city_name_list:
                cluster_num = str(dataframe['player_cluster_dbscan'].iloc[idx])
                if cluster_to_name_dict[cluster_num] == None:
                    city_name = dataframe['city_province'].iloc[idx].split(',')[0]
                    # changing Regina and Brooklyn and Montreal
                    name_switch_dict = {"Regina":'Saskatchewan', 'Brooklyn': 'NYC', "Montreal": "Montreal/Ottawa"
                                       }
                    if city_name in name_switch_dict.keys():
                        city_name = name_switch_dict[city_name]
                    cluster_to_name_dict[cluster_num] = city_name
                counter +=1 
    dataframe['Cluster_name'] = dataframe['player_cluster_dbscan'].astype('str').replace(cluster_to_name_dict)
    return dataframe

def make_points_to_shape(gdf):
    cluster_nums = sorted(gdf['player_cluster_dbscan'].unique())
    cluster_name = []
    convex_hulls = []
    area = []
    for cluster in cluster_nums:
        mini_gdf = gdf[gdf['player_cluster_dbscan']==cluster]
        multipoint = MultiPoint([x for x in mini_gdf['geometry']])
        convex_hull = multipoint.convex_hull
        cluster_name.append(mini_gdf['Cluster_name'].iloc[0])
        convex_hulls.append(convex_hull)
        area.append(convex_hull.area)
    cluster_dict = {'Cluster_num':cluster_nums, 'Cluster_name':cluster_name, 
                    'geometry':convex_hulls, 'Area':area}
    cluster_shape_gdf = gpd.GeoDataFrame(cluster_dict, crs=4326) # this projection uses meters i think
    cluster_shape_gdf_no_lines = cluster_shape_gdf[cluster_shape_gdf['Area']>0]
    return cluster_shape_gdf_no_lines
