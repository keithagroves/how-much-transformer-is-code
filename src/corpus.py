"""Topically-varied short texts. Topic is spread across many domains on purpose
so that the signal a sentiment classifier must encode is *sentiment*, not subject
matter. Labels are NOT hard-coded here: the small model (ministral-3:3b) assigns
them, because it is the ground-truth predictor we are trying to replicate.
"""

TEXTS = [
    # tech / gadgets
    "The battery lasts all day and it barely weighs anything.",
    "The screen cracked after a week and support never replied.",
    "It's a phone. It makes calls and has a screen.",
    "Setup took ten minutes and everything just worked.",
    "The app crashes every time I open the camera.",
    "The specs are average for the price bracket.",
    # restaurants / food
    "Best meal I've had all year, we're already booking again.",
    "Cold fries, rude waiter, and a forty minute wait.",
    "The cafe is on the corner of Fifth and Main.",
    "Every dish came out perfectly seasoned and warm.",
    "I found a hair in my soup and lost my appetite.",
    "They serve lunch from eleven to three on weekdays.",
    # weather / nature
    "What a gorgeous clear morning for a hike.",
    "The storm flooded the basement and ruined the carpet.",
    "It is partly cloudy with a light breeze today.",
    "The sunset over the ridge left everyone speechless.",
    "Three days of gray drizzle and I'm exhausted.",
    "Temperatures will hover around sixty this afternoon.",
    # travel
    "The hotel upgraded us for free and the view was unreal.",
    "Our flight was cancelled and we slept on the floor.",
    "The train departs from platform four every hour.",
    "Locals were so welcoming we didn't want to leave.",
    "The tour was overpriced and the guide seemed bored.",
    "The museum is a short walk from the station.",
    # work / office
    "My manager praised the launch in front of the whole team.",
    "Another pointless meeting that could have been an email.",
    "The report is due by end of business Friday.",
    "I finally got the promotion I've been working toward.",
    "They cut our budget again with no explanation.",
    "The office has four floors and a small cafeteria.",
    # movies / entertainment
    "Stunning visuals and a script that kept me guessing.",
    "Two hours of my life I will never get back.",
    "The film runs about a hundred and forty minutes.",
    "I laughed and cried, easily my favorite of the year.",
    "The plot made no sense and the acting was wooden.",
    "It's a sequel released last spring.",
    # relationships / people
    "She surprised me with tickets and I couldn't stop smiling.",
    "He forgot my birthday for the third year running.",
    "We are meeting for coffee sometime next week.",
    "Their kindness during the move meant the world to me.",
    "The argument left both of us cold for days.",
    "My cousin lives two blocks from here.",
    # sports / fitness
    "We came back from ten points down to win it all.",
    "The team collapsed in the final minutes, heartbreaking.",
    "The match starts at seven on Saturday evening.",
    "Crossing that finish line was the proudest moment of my life.",
    "Injured again, and the season just started, so frustrating.",
    "The gym has a pool and a running track.",
    # shopping / products
    "Arrived early, beautifully packaged, exactly as described.",
    "Wrong size, torn box, and no return label included.",
    "The package weighs about two kilograms.",
    "This jacket is warm, stylish, and worth every penny.",
    "The zipper broke the first time I wore it.",
    "It comes in three colors and two sizes.",
    # home / misc
    "The new couch makes the whole room feel cozy.",
    "The plumber overcharged us and left a mess.",
    "The thermostat is set to sixty-eight degrees.",
    "Fresh paint and clean floors, the place feels brand new.",
    "The neighbors blast music until two every night.",
    "There are two bedrooms and one bathroom.",
    # education / learning
    "The professor made a hard topic genuinely exciting.",
    "The lecture dragged on and I learned nothing.",
    "The class meets on Tuesdays and Thursdays.",
    "I aced the exam after weeks of studying, so relieved.",
    "The textbook is dense, dull, and full of errors.",
    "The syllabus lists twelve chapters in total.",
    # health / service
    "The nurse was gentle and put me completely at ease.",
    "Two hours in the waiting room and still no answers.",
    "The clinic opens at eight and closes at five.",
    "The treatment worked and I feel like myself again.",
    "They lost my results and made me redo every test.",
    "The pharmacy is next to the main entrance.",
]
