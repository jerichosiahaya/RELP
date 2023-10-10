import requests
import json
import datetime
import logging


# Set the endpoint URL and authentication credentials
apiurl = ""
username = ""
password = ""

# Set the initial date and page number
date = "2023-09-05"
page = 1

# Set the goals for collecting data
goal_site_names = "LESTARI"
goal_data_count = 50

# Initialize the data lists
data_lestari = []
data_non_lestari = []

# Loop until the goals are met for both site names
while len(data_lestari) < goal_data_count or len(data_non_lestari) < goal_data_count:
    try:
        # Send a POST request to the endpoint with the date and page number
        response = requests.post(apiurl, auth=(username, password), params={"date": date, "page": page})

        if response.status_code == 200:
            # Parse the response JSON data
            response_data = json.loads(response.text)
            total_page = response_data["TotalPage"]
            item_list = response_data["ItemList"]
            # Loop through the item list and collect the required properties
            for item in item_list:
                
                url = item["Url"]
                site_name = item["SiteName"]
                title = item["Title"]
                description = item["Description"]
                content = item["Content"]
                tags = item["Tag"]
                # Append the data to the corresponding list based on the SiteName property
                if site_name == goal_site_names and len(data_lestari) < goal_data_count:
                    data_lestari.append({
                        "Url": url, 
                        "SiteName": site_name, 
                        "Title": title, 
                        "Description": description, 
                        "Content": content, 
                        "Tag": tags
                    })
                elif len(data_non_lestari) < goal_data_count:
                    data_non_lestari.append({
                        "Url": url, 
                        "SiteName": site_name, 
                        "Title": title, 
                        "Description": description, 
                        "Content": content, 
                        "Tag": tags
                    })
                # Log the data item
            
                # Break the loop if the goals are met for both site names
                if len(data_lestari) >= goal_data_count and len(data_non_lestari) >= goal_data_count:
                    break
            # Increment the page number if it's not the last page
            if page < total_page:
                page += 1
            # Otherwise, reset the page number and increment the date by one day
            else:
                page = 1
                date = (datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            
            # print total data_lestari and data_non_lestari
            progress = page / total_page * 100
            print("Date: {}, Progress: {}, Total data: Lestari: {}, Non Lestari: {}".format(date, progress, len(data_lestari), len(data_non_lestari)))
        else:
            # Log the error response
            logging.error(f"Error: {response.text}")
    except Exception as e:
        # Log the exception
        logging.exception(e)

# Dump the data into two JSON files
with open("LESTARI_VALIDATION.json", "w") as f:
    json.dump(data_lestari, f)
with open("NON_LESTARI_VALIDATION.json", "w") as f:
    json.dump(data_non_lestari, f)