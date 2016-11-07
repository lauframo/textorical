from app import db
from app import Text

def get_training_set(year):
  return Text.query.filter_by(period_start_year = year).filter_by(data_set = "train").limit(20).all()

training_collection = get_training_set(1500) + get_training_set(1600) + get_training_set(1700) + get_training_set(1800) +get_training_set(1900)


print(len(training_collection))

for
