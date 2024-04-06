## AI grade guesser
Used this project to learn about neural networks.

Implemented a way for a neural network to guess what a student will be graded based on different inputs (like study time or parents education).
Data is from a [UCI dataset](https://archive.ics.uci.edu/dataset/320/student+performance)

I followed Andrej Karpathy´s [video](https://www.youtube.com/watch?v=VMj-3S1tku0) on micrograd to do this. 
He built i binary classification neural network from scratch. As my data had 0-20 different grade outputs i changed the code a bit so it implemented softmax. Then i could find a probability for each grade based on the inputs. The grade with the highest probability is choosen as the right grade.
By backpropagating on the loss function i could tune the weights of each neuron in the model (MLP). 

﻿![billede](https://github.com/gbalslevn/gradeGuesser/assets/97167089/b62499eb-187a-4585-81be-264c9236076b)

### Demo

Quickly implemented a GUI to visualize the questions and result. 

![Click here for video-demo](https://github.com/gbalslevn/gradeGuesser/assets/97167089/4f475fca-09b6-463e-bb61-9241a39a6b0e)


I could improve understandability of the answer options but i didnt focus on that. You can read what the different answer options mean on the [UCI](https://archive.ics.uci.edu/dataset/320/student+performance).



