# NCF Model created from Boardgame Geek Data

## Description  
This project explores the creation of a recommendation model using a neural collaborative filtering (NCF) architecture from users' reviews of board games. This project focuses on practicing deep learning with PyTorch.  

NCF's true goal is to produce recommendations from implicit data, not a firm rating; however, we can still leverage the idea of embedding users and items from its design to predict what a user is likely to rate a game.  



## Data
[Data Source](https://www.kaggle.com/datasets/jvanelteren/boardgamegeek-reviews)  

## Based on models discussed here:  
[Recomeder-systems-with-pytorch](https://pureai.substack.com/p/recommender-systems-with-pytorch)  
[Recommendation System with Deep Learning and PyTorch – Hagay Lupesko, Facebook. YouTube]( https://www.youtube.com/watch?v=cqnrFrF3nJ8)

## Notes
To train the model yourself, you can run the board_game_rec.py. Note that this was set up to run with CUDA, so ensure it is installed on your system before running it. The notebook is there solely for final model testing. Please note that this process requires substantial data and may take some time to run. You will need to get the data from Kaggle to run this yourself.
