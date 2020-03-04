from app import app
import os, math
from flask import render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from flaskext.mysql import MySQL
import csv, json, string
from xml.dom import minidom
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pandas import DataFrame

mysql = MySQL()
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'skripsi'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)
symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
tp = []

@app.route('/')
@app.route('/index')
def index():
    return render_template('home.html')

@app.route('/news')
def news():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM news"
    cursor.execute(sql)
    result = cursor.fetchall()
    conn.close()
    return render_template('news.html',news=result)

@app.route('/createNews', methods=["POST"])
def createNews():
    title = request.form['title']
    desc = request.form['desc']
    source = request.form['source']
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "INSERT INTO news (judul, isi, sumber) VALUES (%s, %s, %s)"
    t = (title, desc, source)
    cursor.execute(sql, t)
    conn.commit()
    return redirect(url_for('news'))

@app.route('/updateNews', methods=["POST"])
def updateNews():
    id_news = request.form['id_news']
    title = request.form['utitle']
    desc = request.form['udesc']
    source = request.form['usource']
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "UPDATE news SET judul=%s, isi=%s, sumber=%s WHERE id_news=%s"
    t = (title, desc, source,id_news)
    cursor.execute(sql, t)
    conn.commit()
    return redirect(url_for('news'))

@app.route('/deleteNews/<string:id_news>', methods=['GET'])
def deleteNews(id_news):
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "DELETE FROM news WHERE id_news=%s"
    t = (id_news)
    cursor.execute(sql, t)
    conn.commit()
    return redirect(url_for('news'))

#upload
ALLOWED_EXTENSION=set(['csv','json','xml'])
app.config['UPLOAD_FOLDER'] = 'uploads'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSION

@app.route('/importNews', methods=['GET','POST'])
def importNews():
    if request.method == 'POST':
        file = request.files['file']

        if 'file' not in request.files:
            return redirect(request.url)

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            ext = str(file.filename)
            filename=secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            conn = mysql.connect()
            cursor = conn.cursor()

            if ext.rsplit('.',1)[1] == 'csv':
                csv_data = csv.reader(open(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
                for row in csv_data:
                    t = str(row).strip('[]').strip("'")
                    b = t.rsplit(";")
                    sql = "INSERT INTO news (judul, isi, sumber) VALUES (%s, %s, %s)"
                    cursor.execute(sql, b)
            if ext.rsplit('.',1)[1] == 'json':
                json_data = json.load(open(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
                for row in json_data:
                    judul = row['judul']
                    sumber = row['sumber']
                    isi = row['isi']
                    sql = "INSERT INTO news (judul, isi, sumber) VALUES (%s, %s, %s)"
                    t = (judul, isi, sumber)
                    cursor.execute(sql, t)
            if ext.rsplit('.', 1)[1] == 'xml':
                xml_data = minidom.parse(open(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
                berita = xml_data.getElementsByTagName('berita')
                for row in berita:
                    sumber = row.getElementsByTagName('sumber')[0]
                    judul = row.getElementsByTagName('judul')[0]
                    isi = row.getElementsByTagName('isi')[0]
                    input1 = sumber.firstChild.data
                    input2 = judul.firstChild.data
                    input3 = isi.firstChild.data
                    sql = "INSERT INTO news (judul, isi, sumber) VALUES (%s, %s, %s)"
                    t = (input2, input3, input1)
                    cursor.execute(sql, t)
            conn.commit()
            cursor.close()
    return redirect(url_for('news'))

def stemming(listToStr):
    factorystem = StemmerFactory()
    stemmer = factorystem.create_stemmer()
    stemming = stemmer.stem(listToStr)
    return stemming

@app.route('/texpre')
def textpre():
    conn = mysql.connect()
    cursor = conn.cursor()
    if not tp:
        sql = "SELECT * FROM news"
        cursor.execute(sql)
        result = cursor.fetchall()
        for row in result:
            # casefolding
            # mengubah text menjadi lowercase
            isi = str(row[2])
            lowercase = isi.lower()
            # menghapus tanda baca
            translator = str.maketrans('', '', string.punctuation)
            delpunctuation = lowercase.replace(".", " ").translate(translator)
            # Filtering
            # factory = StopWordRemoverFactory()
            # stopword = factory.create_stop_word_remover()
            # stop = stopword.remove(delpunctuation)
            stopwords = [line.rstrip() for line in open('uploads/stopword.txt')]
            cf = delpunctuation.split()
            stop = [a for a in cf if a not in stopwords]
            listToStr = ' '.join([str(elem) for elem in stop])
            # Stemming
            st = stemming(listToStr)
            #tokenizing
            token = st.split()
            if (row[0], row[1], row[3], delpunctuation, listToStr, st, token) not in tp:
                tp.append((row[0], row[1], row[3], delpunctuation, listToStr, st, token))
        sql = "SELECT * FROM word"
        cursor.execute(sql)
        word = cursor.fetchall()
        if not word:
            for row in tp:
                id_news = row[0]
                token = row[6]
                tktodb = []
                for tk in token:
                    if tk not in tktodb:
                        tktodb.append(tk)
                for row in tktodb:
                    sql = "INSERT INTO word (word, id_news) VALUES(%s, %s)"
                    t = (row, id_news)
                    cursor.execute(sql, t)
        else:
            sql = "DELETE FROM word"
            cursor.execute(sql)
            for row in tp:
                id_news = row[0]
                token = row[6]
                tktodb = []
                for tk in token:
                    if tk not in tktodb:
                        tktodb.append(tk)
                for row in tktodb:
                    sql = "INSERT INTO word (word, id_news) VALUES(%s, %s)"
                    t = (row, id_news)
                    cursor.execute(sql, t)
    conn.commit()
    cursor.close()
    return render_template('textpreprocessing.html', textpre=tp)

@app.route('/tfidf')
def tfidf():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM word"
    cursor.execute(sql)
    result = cursor.fetchall()
    word = []
    wordset = []
    wd= []
    idf = []
    finaltfidf = []
    for row in result:
        word.append(row[1])
    for row in word:
        if row not in wordset:
            wordset.append(row)
    for row in tp:
        id_news = row[0]
        isi = str(row[5])
        isi = isi.split(" ")
        worddict = dict.fromkeys(wordset, 0)
        for word in isi:
            if word in worddict:
                worddict[word] += 1
        wd.append((id_news, worddict))
        idf.append(worddict)
    idfword = computeIDF(idf)
    for row in wd:
        wordtfidf = computeTFIDF(row[1], idfword)
        finaltfidf.append((row[0], wordtfidf))
    sqltf = "SELECT * FROM tf_idf"
    cursor.execute(sqltf)
    result2 = cursor.fetchall()
    if not result2:
        for row in result:
                idword = row[0]
                idnews = row[2]
                sql2 = "INSERT INTO tf_idf (id_word, id_news) VALUES(%s, %s)"
                t2 = (idword, idnews)
                cursor.execute(sql2, t2)
    else:
        deletetfidf = "DELETE FROM tf_idf"
        cursor.execute(deletetfidf)
        for row in result:
                idword = row[0]
                idnews = row[2]
                sql2 = "INSERT INTO tf_idf (id_word, id_news) VALUES(%s, %s)"
                t2 = (idword, idnews)
                cursor.execute(sql2, t2)
    for row in finaltfidf:
        idnews = row[0]
        tf_idf = row[1]
        for word, val in tf_idf.items():
            sql3 = "SELECT id_word FROM word WHERE word=%s AND id_news=%s"
            t3 = (word, idnews)
            cursor.execute(sql3, t3)
            id_word = cursor.fetchall()
            if len(id_word) > 0:
                id_word = str(id_word)
                id_word = id_word.rsplit('((', 1)[1]
                id_word = id_word.split(',)')
                sql4 = "UPDATE tf_idf SET tf_idf=%s WHERE id_news=%s AND id_word=%s"
                t4 = (val, idnews, id_word[0])
                cursor.execute(sql4, t4)
    sql5 = "SELECT * FROM news"
    cursor.execute(sql5)
    result5 = cursor.fetchall()
    sql6 = "SELECT t.id_word, t.id_news, t.tf_idf, w.word FROM `tf_idf`as t inner join word as w on t.id_word=w.id_word"
    cursor.execute(sql6)
    result6 = cursor.fetchall()
    conn.commit()
    cursor.close()
    return render_template('tfidf.html', tfidf=result5, list=result6)

def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict

def computeIDF(docList):
    N = len(docList)
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N/float(val))

    return idfDict

def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf

@app.route('/getKmeans')
def getKmeans():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM news"
    cursor.execute(sql)
    news = cursor.fetchall()
    data = []
    for row in news:
        id_news = row[0]
        sql = "SELECT sum(tf_idf) FROM `tf_idf` WHERE id_news=%s"
        t = (id_news)
        cursor.execute(sql, t)
        sumtfidf = cursor.fetchone()
        sumtfidf = str(sumtfidf).replace("(", "").replace(")", "").replace(",", "")
        data.append(float(sumtfidf))
    df = DataFrame(data, columns=['x'])
    print(df)
    wcss = []
    for i in range(1, 4):
        kmeans = KMeans(n_clusters=i, init = 'k-means++', random_state=0)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
    # plt.plot(range(1, 4), wcss)
    # plt.title('Metode Elbow')
    # plt.xlabel('Jumlah cluster')
    # plt.ylabel('WCSS')
    # plt.show()
    kmeans = KMeans(n_clusters=2, init = 'k-means++', random_state=0)
    x_kmeans = kmeans.fit_predict(df)
    y_kmeans = kmeans.fit_transform(df)
    c = kmeans.cluster_centers_
    print(y_kmeans)
    print(x_kmeans)
    print(c)
    return render_template('kmeans.html')