# Disaster Response Analysis

## Introduction
Heavy rainfall began Friday, August 12, 2016 in Louisiana, with areas reaching almost 3 feet of rain water, causing local rivers to reach historic flood levels. Thousands were forced to evacuate, there are at least 13 dead, and many reported missing. Areas in South Louisiana in and around Lafayette and Baton Rouge were affected most.

Twenty parishes in Louisiana—Acadia, Ascension, East Baton Rouge, Livingston, Avoyelles, Evangeline, East Feliciana, West Feliciana, Iberia, Iberville, Jefferson Davis, Lafayette, Pointe Coupee, St. Helena, St. Landry, St. Martin, St. Tammany, Washington, Tangipahoa, and Vermillion—were declared major federal disaster areas.

Watson, LA—about 20 miles northeast of Baton Rouge—experienced 31.39 inches of rain, White Bayou, LA saw 26.14 inches. Livingston ended up with 25.52 inches. Baton Rouge received over 19 inches.

## Dataset description 

This dataset is picked up from kaggle https://www.kaggle.com/datasets/rahultp97/louisiana-flood-2016 .
Dataset special parameters.

- Image_ID - a unique id for each image.
    - Note: for each before/after the flood image there is a corresponding during the flood image, eg: 3005.png is an image taken before/after the flood and corresponding to that image there is 3005_0.png image which was taken during the flood and the *_0.png implies that the area shown in this image was not flooded.
- Normal - Indicate whether the image is before/after the flood or during the flood.
    - 1 -> the image was taken before/after the flood.
    - 0 -> the image was taken during the flood.
- Flooded - Indicate whether the image contains flooded regions.
   -  1 -> Flooded
   -  0 -> Not flooded.

## Analysis Technique
- Exploratory data analysis
- Feature extraction
- Flood and Non-Flood classification model
- Neural network
- MobileNetV2
- Model testing
- Model Evaluation
- Prediction Summary

## Background analysis
Flooding began in earnest on August 12. On August 13, a flash flood emergency was issued for areas along the Amite and Comite rivers.
The Amite River crested at nearly 5 ft (1.5 m) above the previous record in Denham Springs. Nearly one-third of all homes—approximately 15,000 structures—in Ascension Parish were flooded after a levee along the Amite River was overtopped.
The widespread flooding stranded tens of thousands of people in their homes and vehicles. At least 30,000 people were rescued by local law enforcement, firefighters, the Louisiana National Guard, the Coast Guard and fellow residents, from submerged vehicles and flooded homes.

## Methodologies
### Exploratory data analysis
### Feature extraction
### Flood and Non-Flood classification model
### Neural network
### MobileNetV2
### Model testing
### Model Evaluation
### Prediction Summary

## Conclusion

## References
[1] https://disasterresponse.maps.arcgis.com/apps/StorytellingSwipe/index.html?appid=2e499ec7eb784237bd70fb16ae0f5dcf#
[2] http://louisianaview.org/2016/08/historic-louisiana-floods-august-2016/
[3] https://geodesy.noaa.gov/storm_archive/storms/aug2016_lafloods/index.html#
