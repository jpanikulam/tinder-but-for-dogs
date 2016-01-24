class FB(object):
    def __init__(self):
        self.people_interests = {
            'tess_bianchi': [
                "The Weeknd",
                "The Weekend",
                "Computer Science",
            ],
            'jacob_panikulam': [
                "Ocean's 11",
                "Taylor Swift",
            ],
            'ian_van_stralen': [
                "Barbecue",
                "Robotics"
            ],
        }

    def get_interests(self, name):
        return self.people_interests.get(
            name,
            ['Interests Unknown - Ask about the weather!']
        )
