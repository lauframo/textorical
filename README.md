##Textorical/CuriousCorpus
Textorical evolved to become CuriousCorpus. In this repo you can find the pre-serialized models, to see updated code and find out more please go to [CuriousCorpus](https://github.com/npentella/CuriousCorpus)

## dependencies
  * bssdb3
  * autoenv==1.0.0
  * Flask==0.10.0
  * Flask-SQLAlchemy==2.1
  * Flask-Migrate==1.8.0
  * gutenberg==0.4.2
  * alembic==0.8.8

## Environment configuration:


1. activate your environment
 ```{r, engine='bash'}
   $ source env/bin/activate
 ```

2. install all dependencies
 ```{r, engine='bash'}
 $ pip install -r requirements.txt
 ```


3. start virtual environment
 ```{r, engine='bash'}
   $ echo "source `which activate.sh`" >> ~/.bashrc
   $ source ~/.bashrc
 ```

4. create database
 ```{r, engine='bash'}
   $psql
   # create database textorical;
   CREATE DATABASE
   # \q
 ```

5. run app_settings variable
```
$ export APP_SETTINGS="config.DevelopmentConfig"
```

6. run database_url variable
 ```{r, engine='bash'}
   $ export DATABASE_URL="postgresql://localhost/wordcount_dev"
 ```

7. create migrations folder
  ```{r, engine='bash'}
  $ python manage.py db init
  ```
  * create first migration
  ```{r, engine='bash'}
  $ python manage.py db migrate
  ```
  * run the migration
  ```{r, engine='bash'}
  $ python manage.py db upgrade
  ```

8. seed database
  ```{r, engine='bash'}
  $ python seeds.py
  ```
