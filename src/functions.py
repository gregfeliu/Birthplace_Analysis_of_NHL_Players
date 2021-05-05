import requests
import pandas as pd
import pickle
import geopandas as gpd
from shapely.ops import nearest_points
from shapely.geometry import LineString
import scipy.stats as st
from clustergram import Clustergram
import urbangrammar_graphics as ugg
import seaborn as sns
import matplotlib.pyplot as plt


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
#     player_list = [team_roster_json['roster'][x]['person'] for x in range(len(team_roster_json['roster']))]
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
def fit_plot_clustergram(scaled_data):
    cgram = Clustergram(range(1, 10), n_init=1000, verbose=False)
    cgram.fit(scaled_data)
    ax = cgram.plot(
    figsize=(10, 8),
    line_style=dict(color=ugg.COLORS[1]),
    cluster_style={"color": ugg.COLORS[2]}
    )
    ax.yaxis.grid(False)
    sns.despine(offset=10)
    return cgram

def evaluate_num_of_clusters(clustergram):  
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

##### Plotting the clusters #####
def plot_clusters(us_can_gdf, clustered_gdf, col_name_to_plot: str):
    ax = us_can_gdf.plot(figsize = (12, 12), facecolor='white', edgecolor='grey')
    # the mis-colored dots in the middle of the larger clusters are the centroids of those clusters
    clustered_gdf.plot(column=col_name_to_plot, cmap='tab20b', ax=ax, figsize = (12, 12))
    return ax