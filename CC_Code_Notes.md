### Notes on 08/05/2024:
* I changed the CC_rotate function to use a built-in rotation found in Scipy which I just found out about. Having this function within Scipy is really nice because since I incorporated it into the CC_Rotate algorithm, not much has changed but at least I know that the rotation itself isn't the issue. However, it still seems like the rotation function isn't working as optimally as it should, but I have been hitting my head against the wall with this so I will try to work with Daniel to figure out why it isn't working.
* One thing that I need to figure out is how to stop CC_Rotation if it makes enough progress or something of the sort, similar to CC_Add. I added a tolerance check to ensure there isn't a convergence but it seems like that is too simple of a check since it tends to stop it way earlier than needed.
* I'm not sure if how I looped the CC_Rotation function/ how I structured it is functional because the time it takes for it to run seems to be very variable dependent on how the data is generated (typically takes between 4 and 5 minutes to run whereas adding vectors takes like 10 seconds). This is definetly something that I need to run through with Daniel.
* I think just running through the code with Daniel may help you find the issues that are popping up with the rotation function.

* Here is something that I have noticed, even though the Vector Rotation isn't working currently, something that it has demonstrated is that vector rotation, or some other method entirely, is much more "optimal" than adding a vector each time a point falls out.

### Notes on 08/12/2024:
* I spent most of the day today trying to figure out the best way to animate each of the iterations of CC_Rotate as a gif so that we can see if the cone is actually rotating.

* I have never done something like this before so it took some researching and thinking to find the best way to go about animating this. After a lot of trial and error, looking into packages and seeing whether or not they would work for my needs and messing around with frame_skipping and frame duration I got it working. Lucky for us, it does seem like the cone is rotating properly and the newly refined function is also working properly.
* It does seem like there are a few issues here and there that need to be fixed in the sense of optimality and also how certain parts of the function are written because it seems like one of the vectors gets stuck/doesn't rotate; however, this is really good progress that the code now shows a gif to show how the cone is rotating.
* I also refined my convergence criteria just slightly: I implemented a more sophisticated convergence criterion based on the change in the cone's position or the number of points outside the cone. This will provide a more meaningful way to determine when the algorithm has converged. This is by no means a perfect convergence criteria, and I suggest that we change it sooner rather than later but I can't really thing of a convergence criteria that works and would be easy to implement.
* I will continue thinking about better convergence criteria that I could implement and also try to understand and fix some of the issues that I see within the cone rotation right now. One large issue I see is that the iteration counter in the gif isn't actually working properly, it is just stuck at 27. The second large issue as mentioned is that one vector seems to get stuck during the rotation, or maybe it's because it is almost at mu so the rotations are so small it's hard to see.

### Notes on 08/16/2024:
* Since Daniel and I didn't meet on Wednesday, we pushed the meeting to today at 2:30 on Zoom. I spent time until the meeting trying to debug and got no where. Showed Daniel the animations and everything but there wasn't much he could really see at first glance looking at the code so mainly the meeting was me showing him the animation.
* During the meeting Daniel told me to add the mu vector to the gif so that we could see where mu really was and if the cone was truly rotating around it and why there was a vector that kept getting stuck on it. I did that and we did see that vectors were getting stucks on it.
* After the meeting I was debugging some more before my next meeting at 4 and luckily I found out that I was updating the vectors during the cone rotation which was messing with some of the vectors and completely collapsing them all the way until they hit mu right before the rotation started. With this line taken out, the cone finally started rotating as a whole!
* Additionally, I commented out the line that pushes the vectors back one step at the start of the rotation. When watching the gif, and looking at the final cone, it seemed as though the process/push was hurting the algorithm more than it was helping it.
* Lastly, I pushed the convergence criteria super low to see what changes would happen and the performance really only got better by a super slim margin, this just further verifies that a better convergence criteria is needed. But for right now, the algorithm is working much better than it was before today.

### Notes on 08/19/2024:
* I tried messing around with the convergence criteria and the update_vector function in the code since I felt like what I have right now wasn't really optimal or consistent; however, the only thing that I got working was the update_vector function. The updated vector update function takes into account the distribution of points outside the cone, allowing for more targeted updates.
* I tried to use the same concept with the convergence criteria but didn't have any luck with that, I am going to try again before meeting with Daniel because I really feel like that convergence criteria needs to be further refined to properly allow for CC_Rotate to converge to the actual shape of the data without taking too long.

### Notes on 08/21/2024:
* I took a stab at trying a density-based vector update approach, it seems to be just slightly more optimal than the previous method, but, it still has a lot of work to be done.
* After meeting with Daniel, he had mentioned that building upon what I did with the density-based approach, I cna do a distance-based approach where I update the vector that has the greatest amount of distance between itself and the closest point instead of cycling between vectors aimlessly when updating.
* Refining the code to do this wasn't all too hard since I had already done the density-based approach. Once I did this I felt like adding planes between the vectors would be nice to have to better visualize the boundries of the cone so I refined the plots to create these boundries as well as added a green mu line into the interactive plots.
    * I'm glad I did this, as it was nice to see the complete shape of the cone.
* Finally I cleaned up my code, refined function names, put all my functions into a .py file that I run at the start of my notebook, created this markdown file for all my notes, and refined my notebook summary.
* Sometime next week I will push my folder with all my stuff from this rotation to GitHub for Mike to use when I pass the torch to him. I will also make my rotation presentation and include it in the folder for Mike or whoever else to use in the future.

### Notes on 09/03/2024:
* Finished my PPT Presentation for the Rotation Project, if you would like a copy of the presentation please feel free to e-mail me at Ounsinegad@wisc.edu
* After everything was completed I created a repository for this project and uploaded my code and associated files for future use by any students, faculty, or staff who may work on this project, including Daniel. The link to the GitHub repository is https://github.com/Aurodexdis/ConeCollapseNMF

### Notes on 09/06/2024:
* After meeting w/ Daniel and Mike, Daniel told me that mu was supposed to be the center of the data, not the center of the nonnegative orthant, this was a mistake on my end. This was an easy fix though, I went in and changed where mu was calculated in each function and how it was plotted and then reran the main code to get new plots for my rotation presentation.
* Refined my PPT Presentation for the Rotation Project with the new edits.
* After everything was completed I pushed my new code and notes to the GitHub repo.
