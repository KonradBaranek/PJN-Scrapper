import urllib.request
import codecs
from bs4 import BeautifulSoup

directors = []
actors = []
genres = []
countries = []

def scrapp_filmweb_page(page):
    with urllib.request.urlopen(
            "https://www.filmweb.pl/films/search?orderBy=popularity&descending=true&page=" + str(page)) as response:
        soup = BeautifulSoup(response.read(), "html.parser")

        films = soup.find_all('div', class_='filmPreview__card')
        dicts = []
        for film in films:
            dict = {}
            dict['title'] = film.find('h3', class_='filmPreview__title').text
            dict['year'] = film.find('span', class_='filmPreview__year').text
            if film.find('span', class_='rateBox__rate') :
                dict['rate'] = (film.find('span', class_='rateBox__rate').text)
            else:
                dict['rate'] = None
            dict['want_to_see'] = film.find('span', class_='wantToSee__count').text if film.find('span', class_='wantToSee__count') else None
            dict['votes'] = film.find('span', class_='rateBox__votes').text[:-7]  if film.find('span', class_='rateBox__votes') else None
            dict['genres'] = list(map(lambda x: x.text, film.find('div', class_='filmPreview__info--genres').find_all('a') if film.find('div', class_='filmPreview__info--genres') else []))
            dict['director'] = film.find('div', class_='filmPreview__info--directors').find('a').text if film.find('div', class_='filmPreview__info--directors') else None
            dict['countries'] = film.find('div', class_='filmPreview__info--countries').find('a').text if film.find('div', class_='filmPreview__info--countries') else None
            dict['cast'] =  film.find('div', class_='filmPreview__info--cast')
            if dict['cast']:
                dict['cast'] = list(map(lambda x: x.text, dict['cast'].find_all('a')))
            else:
                dict['cast'] = None
            dicts.append(dict)
        with codecs.open('movies.data', "a+", encoding='utf-8') as movies:
            for dict in dicts:
                if dict['genres'] and dict['director'] and dict['rate'] and dict['cast']:
                    movies.write(str(dict['rate']).replace(",",".")+'\t')
                    movies.write('"{}"\t'.format(str(dict['title'])))
                    movies.write(str(dict['year'])+'\t')
                    movies.write(str(dict['want_to_see']).replace(" ", "")+'\t')
                    movies.write(str(dict['votes'].replace(" ", ""))+'\t')
                    if dict['genres'][0] not in genres:
                        genres.append(dict['genres'][0])
                    movies.write(str(genres.index(dict['genres'][0]))+'\t')
                    if dict['director'] not in directors:
                        directors.append(dict['director'])
                    movies.write(str(directors.index(dict['director']))+'\t')
                    if dict['countries'] not in countries:
                        countries.append(dict['countries'])
                    movies.write(str(countries.index(dict['countries']))+'\t')
                    if dict['cast'][0] not in actors:
                        actors.append(dict['cast'][0])
                    movies.write(str(actors.index(dict['cast'][0])))
                    movies.write(u'\n')
            movies.close()

