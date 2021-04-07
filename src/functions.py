import requests
import pandas as pd
import pickle


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