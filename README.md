# TADAI Camp Music Recommendation Backend 🎵

Welcome to the TADAI Camp Music Recommendation Backend! 🚀 This Flask-based API hosted on AWS EC2 is here to provide you with awesome music recommendations. 🎶

## Overview 📝

TADAI Camp is all about using the power of Machine Learning to recommend music that truly vibes with your taste. 🎧 We combine natural language processing (NLP), song feature extraction, and Principal Component Analysis (PCA) to create a unique map of songs. This means our recommendations go way beyond just matching sound.

## Features 🌟

- 😍 Natural Language Processing (NLP): We analyze song lyrics with sentiment analysis and keyword extraction to understand the emotions behind the songs.

- 🎵 Song Feature Extraction: We extract attributes like song duration, BPM, and MFCC to get a deeper understanding of the music.

- 🤖 Machine Learning: Our secret sauce is Principal Component Analysis (PCA) that helps us identify songs with similar vibes.

- 🎶 Audio Analysis: We use Mel-frequency cepstral coefficients (MFCC) and audio clips to dive deep into the sound characteristics.

- ⚙️ Efficient Backend: We've built this backend with Python and Flask for seamless development.

- 🌐 Cloud-Powered: We leverage AWS for a cost-effective and scalable solution.

- 💰 Future Plans: We're looking at introducing free and premium plans in the future as we gain popularity.

## How to Use 🤔

To get song recommendations, make a GET request using the following query parameters:

- `title`: The title of the song.
- `artist`: The artist of the song.
- `nb_of_recommendations`: The number of recommendations you want.

Example Query:
```
http://13.38.95.183/predict?title=Parfum%20quartier&artist=Jul&nb_of_recommendations=10
```

## Future Endeavors 🚀

The TADAI Camp team is passionate about the intersection of music and AI. We see this project as just the beginning. One exciting avenue we're looking into is the development of generative AI for music composition. 🎵🤖

## Developers 👩‍💻👨‍💻

This project was brought to life by:
- Jean Doutriaux
- Cléophas Fournier
- Nicolas Violot

## Get Groovin'! 🕺💃

Now that you're familiar with our Music Recommendation Backend, give it a try and discover some amazing music recommendations! 🎉 Enjoy the music journey! 🎵🔥
