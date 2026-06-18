# Week 5

This week was insanely hard, and it was something I did not expect to happen. I started implementing text segmentation, first by lines, and that part worked. After that I tried to move on to character segmentation, but I could not get it to work, and I found it extremely difficult to implement.

Receipt images are very noisy, and at this stage I do not have enough knowledge to solve that problem properly. I learned that a fairly straightforward pipeline can still become very difficult, and that solving character segmentation may require its own network and more advanced techniques than I can currently implement.

I spent about 30 hours on the project this week, but I will mark only 10 hours, because I do not have working code to show. The only code change I can point to is the deleted `line_segmentation.py` file.

I also did research on how to change the model so it could solve a similar problem in a better way. Based on that, I decided to implement text recognition instead. The new solution will recognize text from books or PDF text, for example, and I will make the necessary changes in my project. Basically, the project will still solve a similar problem, but with much less noisy data.

## Time Tracking

| Date | Time Spent | Description |
| ----- | ----------- | ----------- |
| 10.6. | 3 h | Implementing line-based text segmentation |
| 11.6. | 4 h | Trying to implement character segmentation |
| 12.6. | 3 h | Researching text recognition as a better approach |
| **Total** | **10 h** | |
