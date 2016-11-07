from app import db

import settings

# DeclarativeBase = declarative_base()


# def db_connect():
#     #   """
#     # Performs database connection using database settings from settings.py.
#     # Returns sqlalchemy engine instance
#     # """
#     return create_engine(URL(**settings.DATABASE))

# def create_texts_table(engine):
#   DeclarativeBase.metadata.create_all(engine)


class Text(db.Model):
    __tablename__ = "texts"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column('title', db.String(), nullable=False, unique=True)
    author = db.Column('author', db.String(), nullable=False)
    author_birth_year = db.Column('author_birth_year', db.Integer, nullable=False)
    period_start_year = db.Column('period_start_year', db.Integer, nullable=False)
    text_content = db.Column('text_content', db.Text, nullable=False)
    gutenberg_id = db.Column('gutenberg_id', db.Integer, nullable=False, unique=True)
    bag_of_words_id = db.Column('bag_of_words_id', db.Integer)
    data_set = db.Column('data_set', db.String())


    def __init__(self, title, author, author_birth_year, period_start_year, text_content, gutenberg_id, bag_of_words_id, data_set):
        self.title = title
        self.author = author
        self.author_birth_year = author_birth_year
        self.period_start_year = period_start_year
        self.text_content = text_content
        self.gutenberg_id = gutenberg_id
        self.bag_of_words_id = bag_of_words_id
        self.data_set = data_set

    def __repr__(self):
        return '<id {}>'.format(self.id)

