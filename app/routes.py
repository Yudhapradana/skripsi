from app import app
import os, math, datetime
from flask import render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from flaskext.mysql import MySQL
import csv, json, string
from xml.dom import minidom
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from flask import jsonify
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from flask_navigation import Navigation

nav = Navigation(app)
mysql = MySQL()
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'skripsi'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)
symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
tp = []
nav.Bar('top', [
    nav.Item('Home', 'index'),
    nav.Item('News', 'news'),
])

@app.route('/')
@app.route('/index')
def index():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM testing ORDER BY query ASC"
    cursor.execute(sql)
    k = cursor.fetchall()
    sql = "SELECT DISTINCT query, typeir FROM testing ORDER BY query ASC"
    cursor.execute(sql)
    query = cursor.fetchall()
    conn.close()
    return render_template('home.html', kluster=k, query=query)

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
    sql = "SELECT * FROM news"
    cursor.execute(sql)
    result = cursor.fetchall()
    if not tp:
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
                for row in token:
                    sql = "INSERT INTO word (word, id_news) VALUES(%s, %s)"
                    t = (row, id_news)
                    cursor.execute(sql, t)
        else:
            sql = "TRUNCATE word"
            cursor.execute(sql)
            for row in tp:
                id_news = row[0]
                token = row[6]
                for row in token:
                    sql = "INSERT INTO word (word, id_news) VALUES(%s, %s)"
                    t = (row, id_news)
                    cursor.execute(sql, t)
    conn.commit()
    cursor.close()
    return render_template('textpreprocessing.html', textpre=tp)

@app.route('/getNews')
def getNews():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM news"
    cursor.execute(sql)
    result = cursor.fetchall()
    conn.close()

    res = []
    for row in result:
        res.append({ "id" : row[0],"judul" : row[1],})

    data = {}
    data["data"] = res
    return jsonify(data)

@app.route('/tfidf')
def tfidf():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql6 = "SELECT t.id_word, t.id_news, t.tf_idf, w.word FROM `tf_idf`as t inner join word as w on t.id_word=w.id_word"
    cursor.execute(sql6)
    result6 = cursor.fetchall()
    sql = "SELECT * FROM news"
    cursor.execute(sql)
    news = cursor.fetchall()
    if not result6:
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
        listberita = []
        for row in news:
            listword = []
            id_news = row[0]
            sql = "SELECT * FROM word WHERE id_news=%s"
            t = (id_news)
            cursor.execute(sql, t)
            word = cursor.fetchall()
            for row in word:
                w = str(row[1])
                listword.append(w)
            listberita.append((id_news, listword))
        for row in listberita:
            id_news = row[0]
            isi = row[1]
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
                    valueword = row[1]
                    idnews = row[2]
                    sql = "SELECT word.word, word.id_news FROM tf_idf inner join word on tf_idf.id_word=word.id_word WHERE word.id_news=%s AND word.word=%s"
                    t = (idnews, valueword)
                    cursor.execute(sql, t)
                    valueword2 = cursor.fetchall()
                    if not valueword2:
                        sql2 = "INSERT INTO tf_idf (id_word, id_news) VALUES(%s, %s)"
                        t2 = (idword, idnews)
                        cursor.execute(sql2, t2)
        else:
            deletetfidf = "TRUNCATE tf_idf"
            cursor.execute(deletetfidf)
            for row in result:
                idword = row[0]
                valueword = row[1]
                idnews = row[2]
                sql = "SELECT word.word, word.id_news FROM tf_idf inner join word on tf_idf.id_word=word.id_word WHERE word.id_news=%s AND word.word=%s"
                t = (idnews, valueword)
                cursor.execute(sql, t)
                valueword2 = cursor.fetchall()
                if not valueword2:
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
        sql6 = "SELECT t.id_word, t.id_news, t.tf_idf, w.word FROM `tf_idf`as t inner join word as w on t.id_word=w.id_word"
        cursor.execute(sql6)
        result6 = cursor.fetchall()
    conn.commit()
    cursor.close()
    return render_template('tfidf.html', tfidf=news, list=result6)

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
        if val < 1:
            idfDict[word] = 0
        else:
            idfDict[word] = (float(math.log10(N/float(val))))+1
    return idfDict

def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf

@app.route('/doc2vec')
def doc2vec():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM news"
    cursor.execute(sql)
    news = cursor.fetchall()
    listberita = []
    idnews = []
    title = []
    for row in news:
        listword = []
        id_news = row[0]
        judul = row[1]
        sql = "SELECT * FROM word WHERE id_news=%s"
        t = (id_news)
        cursor.execute(sql, t)
        word = cursor.fetchall()
        for row in word:
            w = str(row[1])
            listword.append(w)
        listberita.append((id_news, listword))
        idnews.append(id_news)
        title.append(judul)
    data = []
    for row in listberita:
        sentence = ""
        word = row[1]
        for row in word:
            sentence+=row+" "
        data.append(str(sentence))
    tagged_data = [TaggedDocument(words=_d.split(), tags=[str(i)]) for i, _d in enumerate(data)]
    model = Doc2Vec(tagged_data)
    model.save("dv2.model")
    # print(len(model.infer_vector(["4g", "lte"])))
    N = len(model.docvecs)
    vecdoc = []
    for row in range(N):
        vecdoc.append(model.docvecs[row])
    joinidnewsvecdoc = np.array([idnews, title, vecdoc])
    joinidnewsvecdoc_transpose = joinidnewsvecdoc.transpose()
    # array = np.array(vecdoc)
    # kmeans = MiniBatchKMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None).fit(array)
    return render_template('doc2vec.html', result=joinidnewsvecdoc_transpose)

@app.route('/getKmeansTfidf')
def getKmeansTfidf():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM cluster_kmeans WHERE type='tfidf' ORDER BY id ASC"
    cursor.execute(sql)
    kmeans = cursor.fetchall()
    return render_template('kmeans.html', kmeans=kmeans)

@app.route('/checkCluster')
def checkcluster():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM word"
    cursor.execute(sql)
    result = cursor.fetchall()
    word = []
    wordset = []
    for row in result:
        word.append(row[1])
    for row in word:
        if row not in wordset:
            wordset.append(row)
    sql = "SELECT * FROM news"
    cursor.execute(sql)
    news = cursor.fetchall()
    wd = []
    id_news = []
    for row in news:
        id = row[0]
        worddict = dict.fromkeys(wordset, 0)
        wd.append((id, worddict))
        if id not in id_news:
            id_news.append(id)
    valuewd = []
    for row in wd:
        idnews = row[0]
        isi = row[1]
        worddict2 = dict.fromkeys(wordset, 0)
        for word, val in isi.items():
            sql = "SELECT tf_idf FROM tf_idf INNER JOIN word on tf_idf.id_word=word.id_word WHERE tf_idf.id_news=%s AND word.word=%s"
            t = (idnews, word)
            cursor.execute(sql, t)
            valuetfidf = cursor.fetchone()
            valuetfidf = str(valuetfidf).replace("(", "").replace(")", "").replace(",", "")
            if valuetfidf == "None":
                worddict2[word] = float(0)
            else:
                worddict2[word] = float(valuetfidf)
        valuewd.append(worddict2)
    matrix = []
    for row in valuewd:
        value = []
        for word, val in row.items():
            value.append(val)
        matrix.append(value)
    array = np.array(matrix)
    wcss = []
    for i in range(1, 4):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=123)
        kmeans.fit(array)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 4), wcss)
    plt.title('Metode Elbow')
    plt.xlabel('Jumlah cluster')
    plt.ylabel('WCSS')
    plt.show()
    conn.close()
    return redirect(url_for('getKmeansTfidf'))

@app.route('/getClusterTfidf', methods=["POST"])
def getClusterTfidf():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM word"
    cursor.execute(sql)
    result = cursor.fetchall()
    word = []
    wordset = []
    for row in result:
        word.append(row[1])
    for row in word:
        if row not in wordset:
            wordset.append(row)
    sql = "SELECT * FROM news"
    cursor.execute(sql)
    news = cursor.fetchall()
    wd = []
    id_news = []
    for row in news:
        id = row[0]
        worddict = dict.fromkeys(wordset, 0)
        wd.append((id, worddict))
        if id not in id_news:
            id_news.append(id)
    valuewd = []
    for row in wd:
        idnews = row[0]
        isi = row[1]
        worddict2 = dict.fromkeys(wordset, 0)
        for word, val in isi.items():
            sql = "SELECT tf_idf FROM tf_idf INNER JOIN word on tf_idf.id_word=word.id_word WHERE tf_idf.id_news=%s AND word.word=%s"
            t = (idnews, word)
            cursor.execute(sql, t)
            valuetfidf = cursor.fetchone()
            valuetfidf = str(valuetfidf).replace("(", "").replace(")", "").replace(",", "")
            if valuetfidf == "None":
                worddict2[word] = float(0)
            else:
                worddict2[word] = float(valuetfidf)
        valuewd.append(worddict2)
    matrix = []
    for row in valuewd:
        value = []
        for word, val in row.items():
            value.append(val)
        matrix.append(value)
    array = np.array(matrix)
    total = request.form['total']
    # wcss = []
    # for i in range(1, 4):
    #     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    #     kmeans.fit(array)
    #     wcss.append(kmeans.inertia_)
    # # plt.plot(range(1, 4), wcss)
    # # plt.title('Metode Elbow')
    # # plt.xlabel('Jumlah cluster')
    # # plt.ylabel('WCSS')
    # # plt.show()
    kmeans = MiniBatchKMeans(n_clusters=int(total), init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=123)
    kmeans.fit(array)
    resultcluster = kmeans.labels_
    centroid = kmeans.cluster_centers_
    c = []
    for row in resultcluster:
        c.append(("Cluster "+str(row), 0))
    c2 = []
    for row in c:
        if row not in c2:
            c2.append(row)
    type = "tfidf"
    print(c2)
    for row in c2:
        cluster = row[0]
        datakmeans = "SELECT cluster, total FROM cluster_kmeans WHERE cluster=%s AND total=%s AND type=%s"
        t = (cluster, total, type)
        cursor.execute(datakmeans, t)
        result = cursor.fetchall()
        print(result)
        if not result:
            sql = "INSERT INTO cluster_kmeans(cluster ,total, type) VALUES(%s, %s, %s)"
            t = (cluster, total, type)
            cursor.execute(sql, t)
    idkmeans=[]
    for row in c:
        clus = row[0]
        sql = "SELECT id FROM cluster_kmeans WHERE cluster=%s AND total=%s AND type=%s"
        t = (clus, total, type)
        cursor.execute(sql, t)
        id = cursor.fetchone()
        id = str(id).replace("(", "").replace(")", "").replace(",", "")
        idkmeans.append(id)
    joinidnewsidkmeans = np.array([id_news, idkmeans])
    joinidnewsidkmeans_transpose = joinidnewsidkmeans.transpose()
    for row in joinidnewsidkmeans_transpose:
        text = str(row).replace("['", "").replace("' '", " ").replace("']", "").split(" ")
        id = text[0]
        idc = text[1]
        dataresultkmeans = "SELECT total FROM result_kmeans INNER JOIN cluster_kmeans on result_kmeans.id_cluster=cluster_kmeans.id WHERE result_kmeans.id_news=%s AND total=%s AND type=%s"
        t = (id, total, type)
        cursor.execute(dataresultkmeans, t)
        result = cursor.fetchall()
        totalset = []
        for r in result:
            r = str(r).replace("(", "").replace(")", "").replace(",", "")
            if r not in totalset:
                totalset.append(r)
        if not totalset:
            sql = "INSERT INTO result_kmeans(id_cluster, id_news) VALUES(%s, %s)"
            t = (idc, id)
            cursor.execute(sql, t)
    sql = "SELECT word.id_news, id_word, result_kmeans.id_cluster, cluster_kmeans.cluster FROM `word` INNER JOIN result_kmeans on word.id_news=result_kmeans.id_news INNER JOIN cluster_kmeans on cluster_kmeans.id=result_kmeans.id_cluster WHERE cluster_kmeans.total=%s AND type=%s"
    t = (total, type)
    cursor.execute(sql, t)
    result = cursor.fetchall()
    clustot = []
    for row in result:
        idword = row[1]
        idcluster = row[2]
        idnews = row[0]
        if not clustot:
            sql = "SELECT DISTINCT cluster_kmeans.total FROM `word` INNER JOIN result_kmeans on word.id_news=result_kmeans.id_news INNER JOIN cluster_kmeans on cluster_kmeans.id=result_kmeans.id_cluster WHERE cluster_kmeans.total=%s AND type=%s"
            t = (total, type)
            cursor.execute(sql, t)
            result2 = cursor.fetchall()
            clustot.append(result2)
        if clustot:
            sql = "SELECT centroid_kmeans.id_cluster, centroid_kmeans.id_word, total FROM `centroid_kmeans` INNER JOIN cluster_kmeans on centroid_kmeans.id_cluster=cluster_kmeans.id WHERE centroid_kmeans.id_cluster=%s AND centroid_kmeans.id_word=%s AND cluster_kmeans.total=%s AND type=%s"
            t = (idcluster, idword, total, type)
            cursor.execute(sql, t)
            result3 = cursor.fetchall()
            if not result3:
                sql = "SELECT word FROM word WHERE id_word=%s"
                t = (idword)
                cursor.execute(sql, t)
                result4 = cursor.fetchone()
                valueword = str(result4).replace("(", "").replace(")", "").replace(",", "").replace("'", "")
                if result4:
                    sql = "SELECT word.word, word.id_news FROM centroid_kmeans inner join word on centroid_kmeans.id_word=word.id_word WHERE word.id_news=%s AND word.word=%s AND centroid_kmeans.id_cluster=%s"
                    t = (idnews, valueword, idcluster)
                    cursor.execute(sql, t)
                    result5 = cursor.fetchall()
                    if not result5:
                        sql = "INSERT INTO centroid_kmeans(id_cluster, id_word) VALUES(%s, %s)"
                        t = (idcluster, idword)
                        cursor.execute(sql, t)
    savecentroid = []
    i = 0
    for row in centroid:
        listval = []
        for v in row:
            listval.append(v)
        savecentroid.append((i, listval, wordset))
        i+=1
    for row in savecentroid:
        clus = "Cluster "+str(row[0])
        sql = "SELECT id FROM cluster_kmeans WHERE cluster=%s AND total=%s AND type=%s"
        t = (clus, total, type)
        cursor.execute(sql, t)
        result = cursor.fetchone()
        idc = str(result).replace("(", "").replace(")", "").replace(",", "").replace("'", "")
        joinwordcentroid = np.array([row[1], row[2]])
        joinidnewsidkmeans_transpose = joinwordcentroid.transpose()
        for rows in joinidnewsidkmeans_transpose:
            word = rows[1]
            val = rows[0]
            sql = "SELECT centroid_kmeans.id_word FROM `centroid_kmeans` INNER JOIN cluster_kmeans on centroid_kmeans.id_cluster=cluster_kmeans.id INNER JOIN word on centroid_kmeans.id_word=word.id_word WHERE centroid_kmeans.id_cluster=%s AND word.word=%s"
            t = (idc, word)
            cursor.execute(sql, t)
            result2 = cursor.fetchall()
            for insert in result2:
                idw = str(insert).replace("(", "").replace(")", "").replace(",", "")
                sql = "UPDATE centroid_kmeans SET centroid=%s WHERE id_cluster=%s AND id_word=%s"
                t = (val, idc, idw)
                cursor.execute(sql, t)
    conn.commit()
    cursor.close()
    return redirect(url_for('getKmeansTfidf'))

@app.route('/getKmeansDoc2vec')
def getKmeansDoc2vec():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM cluster_kmeans WHERE type='doc2vec' ORDER BY id ASC"
    cursor.execute(sql)
    kmeans = cursor.fetchall()
    return render_template('kmeans2.html', kmeans=kmeans)

@app.route('/getClusterDoc2vec', methods=["POST"])
def getClusterDoc2vec():
    conn = mysql.connect()
    cursor = conn.cursor()
    total = request.form['total']
    model = Doc2Vec.load("dv2.model")
    sql = "SELECT * FROM news"
    cursor.execute(sql)
    news = cursor.fetchall()
    # listberita = []
    idnews = []
    for row in news:
        # listword = []
        id_news = row[0]
        # sql = "SELECT * FROM word WHERE id_news=%s"
        # t = (id_news)
        # cursor.execute(sql, t)
        # word = cursor.fetchall()
        # for row in word:
        #     w = str(row[1])
        #     listword.append(w)
        # listberita.append((id_news, listword))
        idnews.append(id_news)
    # data = []
    # for row in listberita:
    #     sentence = ""
    #     word = row[1]
    #     for row in word:
    #         sentence += row + " "
    #     data.append(str(sentence))
    # tagged_data = [TaggedDocument(words=_d.split(), tags=[str(i)]) for i, _d in enumerate(data)]
    # model = Doc2Vec(tagged_data)
    # model.save("dv2.model")
    # print(len(model.infer_vector(["4g", "lte"])))
    N = len(model.docvecs)
    vecdoc = []
    for row in range(N):
        vecdoc.append(model.docvecs[row])
    array = np.array(vecdoc)
    kmeans = MiniBatchKMeans(n_clusters=int(total), init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0,
                             random_state=123)
    kmeans.fit(array)
    resultcluster = kmeans.labels_
    centroidcluster = kmeans.cluster_centers_
    c = []
    for row in resultcluster:
        c.append(("Cluster " + str(row), 0))
    c2 = []
    for row in c:
        if row not in c2:
            c2.append(row)
    type = "doc2vec"
    for row in c2:
        cluster = row[0]
        datakmeans = "SELECT cluster, total FROM cluster_kmeans WHERE cluster=%s AND total=%s AND type=%s"
        t = (cluster, total, type)
        cursor.execute(datakmeans, t)
        result = cursor.fetchall()
        if not result:
            sql = "INSERT INTO cluster_kmeans(cluster ,total, type) VALUES(%s, %s, %s)"
            t = (cluster, total, type)
            cursor.execute(sql, t)
    idkmeans = []
    for row in c:
        clus = row[0]
        sql = "SELECT id FROM cluster_kmeans WHERE cluster=%s AND total=%s AND type=%s"
        t = (clus, total, type)
        cursor.execute(sql, t)
        id = cursor.fetchone()
        id = str(id).replace("(", "").replace(")", "").replace(",", "")
        idkmeans.append(id)
    joinidnewsidkmeans = np.array([idnews, idkmeans])
    joinidnewsidkmeans_transpose = joinidnewsidkmeans.transpose()
    for row in joinidnewsidkmeans_transpose:
        text = str(row).replace("['", "").replace("' '", " ").replace("']", "").split(" ")
        id = text[0]
        idc = text[1]
        dataresultkmeans = "SELECT total FROM result_kmeans INNER JOIN cluster_kmeans on result_kmeans.id_cluster=cluster_kmeans.id WHERE result_kmeans.id_news=%s AND total=%s AND type=%s"
        t = (id, total, type)
        cursor.execute(dataresultkmeans, t)
        result = cursor.fetchall()
        totalset = []
        for r in result:
            r = str(r).replace("(", "").replace(")", "").replace(",", "")
            if r not in totalset:
                totalset.append(r)
        if not totalset:
            sql = "INSERT INTO result_kmeans(id_cluster, id_news) VALUES(%s, %s)"
            t = (idc, id)
            cursor.execute(sql, t)
    sql = "SELECT DISTINCT id_cluster FROM centroid_doc2vec INNER JOIN cluster_kmeans on centroid_doc2vec.id_cluster=cluster_kmeans.id WHERE cluster_kmeans.total=%s AND type=%s"
    t = (total, type)
    cursor.execute(sql, t)
    check = cursor.fetchall()
    i = 0
    if not check:
        for row in centroidcluster:
            cluster = "Cluster "+str(i)
            sql = "SELECT id FROM cluster_kmeans WHERE cluster=%s AND total=%s AND type=%s"
            t = (cluster, total, type)
            cursor.execute(sql, t)
            idc = cursor.fetchone()
            idc = str(idc).replace("(", "").replace(")", "").replace(",", "")
            value = row
            for val in value:
                sql = "INSERT INTO centroid_doc2vec(id_cluster, centroid) VALUES(%s,%s)"
                t = (idc, float(val))
                cursor.execute(sql, t)
            i+=1
    conn.commit()
    conn.close()
    return redirect(url_for('getKmeansDoc2vec'))

@app.route('/getClusterNews')
def getClusterNews():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT DISTINCT cluster FROM cluster_kmeans"
    cursor.execute(sql)
    cluster = cursor.fetchall()
    sql = "SELECT DISTINCT total FROM cluster_kmeans"
    cursor.execute(sql)
    total = cursor.fetchall()
    cursor.close()
    return render_template('clusternews.html', cluster=cluster, total=total)

@app.route('/searchClusterNews',methods=['GET','POST'])
def searchClusterNews():
    conn = mysql.connect()
    cursor = conn.cursor()
    cluster = request.form['cluster']
    total = request.form['total']
    type = request.form['type']
    sql = "SELECT k.total, k.cluster, n.judul FROM `result_kmeans` INNER JOIN cluster_kmeans as k on k.id=result_kmeans.id_cluster INNER JOIN news as n on n.id_news=result_kmeans.id_news WHERE k.cluster=%s AND k.total=%s AND k.type=%s ORDER BY k.cluster ASC"
    t = (cluster, total, type)
    cursor.execute(sql, t)
    result = cursor.fetchall()
    sql = "SELECT DISTINCT cluster FROM cluster_kmeans"
    cursor.execute(sql)
    cluster = cursor.fetchall()
    sql = "SELECT DISTINCT total FROM cluster_kmeans"
    cursor.execute(sql)
    total = cursor.fetchall()
    cursor.close()
    return render_template('clusternews.html', result=result, cluster=cluster, total=total)

@app.route('/ir_tfidf_kmeans')
def ir_tfidf_kmeans():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT DISTINCT total FROM cluster_kmeans WHERE type='tfidf'"
    cursor.execute(sql)
    selecttotal = cursor.fetchall()
    return render_template('irs_tfidf_kmeans.html', total=selecttotal)

@app.route('/processTfidfKmeans', methods=['GET','POST'])
def processTfidfKmeans():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT DISTINCT total FROM cluster_kmeans"
    cursor.execute(sql)
    selecttotal = cursor.fetchall()
    query = str(request.form['query'])
    total = request.form['total']
    threshold = request.form['threshold']
    a = datetime.datetime.now().replace(microsecond=0)
    #text preprocessing
    lowercase = query.lower()
    translator = str.maketrans('', '', string.punctuation)
    delpunctuation = lowercase.replace(".", " ").translate(translator)
    stopwords = [line.rstrip() for line in open('uploads/stopword.txt')]
    cf = delpunctuation.split()
    stop = [a for a in cf if a not in stopwords]
    listToStr = ' '.join([str(elem) for elem in stop])
    st = stemming(listToStr)
    token = st.split()
    # end
    dataquery = []
    dataquery.append(('q', token))
    word = []
    wordset = []
    sql = "SELECT * FROM word"
    cursor.execute(sql)
    result = cursor.fetchall()
    for row in result:
        word.append(row[1])
    for row in word:
        if row not in wordset:
            wordset.append(row)
    sql = "SELECT * FROM cluster_kmeans WHERE total=%s ORDER BY cluster"
    t = (total)
    cursor.execute(sql, t)
    datacluster = cursor.fetchall()
    listcentroid = []
    for row in datacluster:
        idcluster = row[0]
        cluster = row[1]
        sql = "SELECT DISTINCT word, centroid FROM `centroid_kmeans` INNER JOIN word on centroid_kmeans.id_word=word.id_word WHERE centroid_kmeans.id_cluster=%s"
        t = (idcluster)
        cursor.execute(sql, t)
        datacentroid = cursor.fetchall()
        worddict = dict.fromkeys(wordset, 0)
        if datacentroid:
            for rows in datacentroid:
                valword = rows[0]
                valcentroid = rows[1]
                if valword in worddict:
                    worddict[valword] = valcentroid
            listcentroid.append((cluster, worddict))
    for row in dataquery:
        token = row[1]
        worddict = dict.fromkeys(wordset, 0)
        for rows in token:
            if rows in worddict:
                worddict[rows]+=1
        listcentroid.append(('q', worddict))
    listworddict = []
    for row in listcentroid:
        worddict = row[1]
        listworddict.append(worddict)
    idfword = computeIDF(listworddict)
    qtfidf = []
    j = 0
    finaltfidf = []
    for row in listcentroid:
        wordtfidf = computeTFIDF(row[1], idfword)
        if row[0] == 'q':
            qtfidf.append((row[0], wordtfidf))
        if j == (len(listcentroid)-1):
            listcentroid.pop(j)
        j+=1
        # finaltfidf.append((row[0], wordtfidf))
    for row in qtfidf:
        listcentroid.append(row)
    computeScalar = computeCrossScalar(listcentroid)
    sumComputeScalar = computeSumScalar(computeScalar)
    lovector = longvector(listcentroid)
    N = len(lovector)
    i = 0
    datalovector = []
    for row in lovector:
        if i == (N-1):
            qlovector = row
        else:
            datalovector.append(row)
        i+=1
    joinsumscalardatalovector = np.array([sumComputeScalar, datalovector])
    joinsumscalardatalovector_transpose = joinsumscalardatalovector.transpose()
    cos = []
    c = []
    i = 0
    for row in joinsumscalardatalovector_transpose:
        sumScalar = row[0]
        longvec = row[1]
        C = sumScalar/(qlovector*longvec)
        cos.append((i, C))
        c.append(C)
        i+=1
    print(c)
    print(cos)
    final = []
    for row in cos:
        cluster = "Cluster "+str(row[0])
        if row[1] > float(threshold):
            sql = "SELECT id FROM cluster_kmeans WHERE cluster=%s AND total=%s"
            t = (cluster, total)
            cursor.execute(sql, t)
            result = cursor.fetchall()
            result = str(result).replace("(", "").replace(")", "").replace(",", "")
            if result:
                sql = "SELECT news.id_news, news.judul, news.isi, news.sumber FROM result_kmeans INNER JOIN news ON result_kmeans.id_news=news.id_news WHERE id_cluster=%s"
                t = (result)
                cursor.execute(sql, t)
                final = cursor.fetchall()
            else:
                print("<script>alert('tes');</script>")
    b = datetime.datetime.now().replace(microsecond=0)
    c = b - a
    c = c.seconds
    # recall
    sql = "SELECT * FROM news WHERE isi LIKE '%" + query + "%'"
    cursor.execute(sql)
    n2 = cursor.fetchall()
    count = 0
    for row in final:
        id = row[0]
        for rows in n2:
            id2 = rows[0]
            if id == id2:
                count += 1
    recall = float(count / len(n2))
    precission = float(count / len(final))
    fscore = float(2 * (recall * precission) / (recall + precission))
    sql = "SELECT * FROM testing WHERE query=%s AND threshold=%s AND typeir=%s"
    t = (token, threshold, "tfidf")
    cursor.execute(sql, t)
    check = cursor.fetchall()
    if not check:
        sql = "INSERT INTO testing(query, threshold, n1, n2, n3, time, recall, precission, fscore, typeir) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        t = (token, threshold, count, len(n2), len(final), c, recall, precission, fscore, "tfidf")
        cursor.execute(sql, t)
    else:
        sql = "UPDATE testing SET n1=%s, n2=%s, n3=%s, time=%s, recall=%s, precission=%s, fscore=%s, WHERE query=%s AND threshold=%s AND typeir=%s"
        t = (count, len(n2), len(final), c, recall, precission, fscore, token, threshold, "tfidf")
        cursor.execute(sql, t)
    conn.commit()
    return render_template('irs_tfidf_kmeans.html', total=selecttotal, result=final)

@app.route('/irs_doc2vec_kmeans')
def irs_doc2vec_kmeans():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT DISTINCT total FROM cluster_kmeans WHERE type='doc2vec'"
    cursor.execute(sql)
    selecttotal = cursor.fetchall()
    conn.close()
    return render_template('irs_doc2vec_kmeans.html', total=selecttotal)

@app.route('/processDoc2vecKmeans', methods=['GET','POST'])
def processDoc2vecKmeans():
    conn = mysql.connect()
    cursor = conn.cursor()
    model = Doc2Vec.load("dv2.model")
    query = str(request.form['query'])
    total = request.form['total']
    threshold = request.form['threshold']
    sql = "SELECT DISTINCT total FROM cluster_kmeans WHERE type='doc2vec'"
    cursor.execute(sql)
    selecttotal = cursor.fetchall()
    # clustering
    sql = "SELECT id_news, cluster FROM result_kmeans INNER JOIN cluster_kmeans on result_kmeans.id_cluster=cluster_kmeans.id WHERE cluster_kmeans.total=%s AND cluster_kmeans.type=%s ORDER BY id_news ASC"
    t = (total, "doc2vec")
    cursor.execute(sql, t)
    idnewscluster_transpose = cursor.fetchall()
    sql = "SELECT id, cluster FROM cluster_kmeans WHERE total=%s AND type=%s ORDER BY cluster ASC"
    cursor.execute(sql, t)
    centroid_doc2vec = cursor.fetchall()
    centroidcluster = []
    for row in centroid_doc2vec:
        listcentroid = []
        sql = "SELECT centroid FROM centroid_doc2vec WHERE id_cluster=%s"
        t = (row[0])
        cursor.execute(sql, t)
        result = cursor.fetchall()
        for row in result:
            val = str(row).replace("(", "").replace(",", "").replace(")", "")
            listcentroid.append(float(val))
        centroidcluster.append(listcentroid)
    # N = len(model.docvecs)
    # vecdoc = []
    # for row in range(N):
    #     vecdoc.append(model.docvecs[row])
    # array = np.array(vecdoc)
    # kmeans = MiniBatchKMeans(n_clusters=int(total), init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0,
    #                          random_state=123)
    # kmeans.fit(array)
    # resultcluster = kmeans.labels_
    # centroidcluster = kmeans.cluster_centers_
    # c = []
    # for row in resultcluster:
    #     c.append("Cluster " + str(row))
    # sql = "SELECT * FROM news"
    # cursor.execute(sql)
    # news = cursor.fetchall()
    # idnews = []
    # for row in news:
    #     id = row[0]
    #     idnews.append(id)
    # idnewscluster = np.array([idnews, c])
    # idnewscluster_transpose = idnewscluster.transpose()
    # print(idnewscluster_transpose)
    #text preprocessing
    a = datetime.datetime.now().replace(microsecond=0)
    lowercase = query.lower()
    translator = str.maketrans('', '', string.punctuation)
    delpunctuation = lowercase.replace(".", " ").translate(translator)
    stopwords = [line.rstrip() for line in open('uploads/stopword.txt')]
    cf = delpunctuation.split()
    stop = [a for a in cf if a not in stopwords]
    listToStr = ' '.join([str(elem) for elem in stop])
    st = stemming(listToStr)
    token = st.split()
    # get vector
    vecquery = model.infer_vector(token, steps=20)
    print(vecquery)
    vectorDocCluster = []
    for row in centroidcluster:
        vectorDocCluster.append(row)
    vectorDocCluster.append(vecquery)
    #computecrossscalar
    resultComputeCrossScalar = []
    for row in centroidcluster:
        val = []
        value = []
        centroid = row
        for row1 in vecquery:
            val.append(row1)
        array = np.array([val, centroid])
        array_transpose = array.transpose()
        for row2 in array_transpose:
            if row2[0] < 0:
                row2[0] = 0
            if row2[1] < 0:
                row2[1] = 0
            score = float(float(row2[0])*float(row2[1]))
            value.append(score)
        resultComputeCrossScalar.append(value)
    #computesumscalar
    sum = []
    for row in resultComputeCrossScalar:
        W = 0
        for rows in row:
            W+=float(rows)
        sum.append(W)
    #longvector
    lvector = []
    for row in vectorDocCluster:
        valpow= []
        vector = row
        for row1 in vector:
            score = math.pow(row1, 2)
            valpow.append(score)
        W = 0
        for row2 in valpow:
            W+=float(row2)
        akar = math.sqrt(W)
        lvector.append(akar)
    N = len(lvector)
    i = 0
    datalvector = []
    for row in lvector:
        if i==(N-1):
            qlvector = row
        else:
            datalvector.append(row)
            i+=1
    joinsumscalardatalvector = np.array([sum, datalvector])
    joinsumscalardatalvector_transpose = joinsumscalardatalvector.transpose()
    c = []
    cos = []
    i = 0
    for row in joinsumscalardatalvector_transpose:
        sume = row[0]
        lvectore = row[1]
        C = sume / (qlvector*lvectore)
        cos.append((i, C))
        c.append(C)
        i+=1
    # valuemax = max(c)
    # for row in cos:
    #     if row[1] == valuemax:
    #         choosecluster = "Cluster "+str(row[0])
    # finalnews = []
    # for row in idnewscluster_transpose:
    #     if row[1] == choosecluster:
    #         finalnews.append(row[0])
    # final = []
    # for row in finalnews:
    #     id = row
    #     sql = "SELECT * FROM news WHERE id_news=%s"
    #     t = (id)
    #     cursor.execute(sql, t)
    #     result = cursor.fetchall()
    #     for rows in result:
    #         final.append((rows[0], rows[1], rows[2], rows[3]))
    choosecluster = []
    print(cos)
    for row in cos:
        if row[1] > float(threshold):
            choosecluster.append("Cluster "+str(row[0]))
    finalnews = []
    for row in idnewscluster_transpose:
        if row[1] in choosecluster:
            finalnews.append(row[0])
    final = []
    for row in finalnews:
        id = row
        sql = "SELECT * FROM news WHERE id_news=%s"
        t = (id)
        cursor.execute(sql, t)
        result = cursor.fetchall()
        for rows in result:
            final.append((rows[0], rows[1], rows[2], rows[3]))
    b = datetime.datetime.now().replace(microsecond=0)
    c = b - a
    c = c.seconds
    # recall
    sql = "SELECT * FROM news WHERE isi LIKE '%" + query + "%'"
    cursor.execute(sql)
    n2 = cursor.fetchall()
    count = 0
    for row in final:
        id = row[0]
        for rows in n2:
            id2 = rows[0]
            if id == id2:
                count += 1
    recall = float(count / len(n2))
    precission = float(count / len(final))
    fscore = float(2 * (recall * precission) / (recall + precission))
    sql = "SELECT * FROM testing WHERE query=%s AND threshold=%s AND typeir=%s"
    t = (token, threshold, "doc2vec")
    cursor.execute(sql, t)
    check = cursor.fetchall()
    if not check:
        sql = "INSERT INTO testing(query, threshold, n1, n2, n3, time, recall, precission, fscore, typeir) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        t = (token, threshold, count, len(n2), len(final), c, recall, precission, fscore, "doc2vec")
        cursor.execute(sql, t)
    else:
        sql = "UPDATE testing SET n1=%s, n2=%s, n3=%s, time=%s, recall=%s, precission=%s, fscore=%s, WHERE query=%s AND threshold=%s AND typeir=%s"
        t = (count, len(n2), len(final), c, recall, precission, fscore, token, threshold, "doc2vec")
        cursor.execute(sql, t)
    conn.commit()
    return render_template('irs_doc2vec_kmeans.html', data=final, total=selecttotal)

@app.route('/ir_nocluster')
def ir_nocluster():
    return render_template('irs_nocluster.html')

@app.route('/processNoCluster', methods=['GET','POST'])
def processNoCluster():
    conn = mysql.connect()
    cursor = conn.cursor()
    query = str(request.form['query'])
    threshold = str(request.form['threshold'])
    # text preprocessing
    a = datetime.datetime.now().replace(microsecond=0)
    lowercase = query.lower()
    translator = str.maketrans('', '', string.punctuation)
    delpunctuation = lowercase.replace(".", " ").translate(translator)
    stopwords = [line.rstrip() for line in open('uploads/stopword.txt')]
    cf = delpunctuation.split()
    stop = [a for a in cf if a not in stopwords]
    listToStr = ' '.join([str(elem) for elem in stop])
    st = stemming(listToStr)
    token = st.split()
    listberita = []
    wd = []
    word = []
    wordset = []
    idf = []
    finaltfidf = []
    sql = "SELECT * FROM news"
    cursor.execute(sql)
    news = cursor.fetchall()
    sql = "SELECT * FROM word"
    cursor.execute(sql)
    result = cursor.fetchall()
    for row in result:
        word.append(row[1])
    for row in word:
        if row not in wordset:
            wordset.append(row)
    for row in news:
        listword = []
        id_news = row[0]
        sql = "SELECT * FROM word WHERE id_news=%s"
        t = (id_news)
        cursor.execute(sql, t)
        word = cursor.fetchall()
        for row in word:
            w = str(row[1])
            listword.append(w)
        listberita.append((id_news, listword))
    listberita.append(("q", token))
    for row in listberita:
        id_news = row[0]
        isi = row[1]
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
    computeScalar = computeCrossScalar(finaltfidf)
    sumComputeScalar = computeSumScalar(computeScalar)
    lovector = longvector(finaltfidf)
    N = len(lovector)
    i = 0
    datalovector = []
    for row in lovector:
        if i == (N - 1):
            qlovector = row
        else:
            datalovector.append(row)
        i += 1
    joinsumscalardatalovector = np.array([sumComputeScalar, datalovector])
    joinsumscalardatalovector_transpose = joinsumscalardatalovector.transpose()
    c = []
    for row in joinsumscalardatalovector_transpose:
        sumScalar = row[0]
        longvec = row[1]
        C = sumScalar / (qlovector * longvec)
        c.append(C)
    sql = "SELECT id_news FROM news"
    cursor.execute(sql)
    idnews= cursor.fetchall()
    joinidnewsc = np.array([idnews, c])
    joinidnewsc_transpose = joinidnewsc.transpose()
    final = []
    print(joinidnewsc_transpose)
    for row in joinidnewsc_transpose:
        id = row[0]
        valuecosine = row[1]
        if valuecosine > float(threshold):
            print(valuecosine)
            sql = "SELECT * FROM news WHERE id_news=%s"
            t = (id)
            cursor.execute(sql, t)
            result = cursor.fetchall()
            for row in result:
                final.append((row[0], row[1], row[2], row[3]))
    b = datetime.datetime.now().replace(microsecond=0)
    c = b-a
    c = c.seconds
    #recall
    sql = "SELECT * FROM news WHERE isi LIKE '%" + query + "%'"
    cursor.execute(sql)
    n2 = cursor.fetchall()
    count = 0
    for row in final:
        id = row[0]
        for rows in n2:
            id2 = rows[0]
            if id == id2:
                count += 1
    recall = float(count / len(n2))
    precission = float(count / len(final))
    fscore = float(2 * (recall * precission) / (recall + precission))
    sql = "SELECT * FROM testing WHERE query=%s AND threshold=%s AND typeir=%s"
    t = (token, threshold, "nocluster")
    cursor.execute(sql, t)
    check = cursor.fetchall()
    if not check:
        sql = "INSERT INTO testing(query, threshold, n1, n2, n3, time, recall, precission, fscore, typeir) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        t = (token, threshold, count, len(n2), len(final), c, recall, precission, fscore, "nocluster")
        cursor.execute(sql, t)
    else:
        sql = "UPDATE testing SET n1=%s, n2=%s, n3=%s, time=%s, recall=%s, precission=%s, fscore=%s, WHERE query=%s AND threshold=%s AND typeir=%s"
        t = (count, len(n2), len(final), c, recall, precission, fscore, token, threshold, "nocluster")
        cursor.execute(sql, t)
    conn.commit()
    return render_template('irs_nocluster.html', result=final)

def computeCrossScalar(tfidf):
    N = len(tfidf)
    i = 0
    Q = []
    doc = []
    for row in tfidf:
        if i == (N-1):
            Q.append(row[1])
        else:
            doc.append(row[1])
        i+=1
    result = []
    for row in doc:
        wdDict = dict.fromkeys(doc[0].keys(), 0)
        for row2 in Q:
            for word, val in row2.items():
                wdDict[word]+=float(val)
        for word, val in row.items():
            wdDict[word]*=float(val)
        result.append(wdDict)
    return result

def computeSumScalar(computeScalar):
    sum = []
    for row in computeScalar:
        W = 0
        for word, val in row.items():
            W+=float(val)
        sum.append(W)
    return sum

def longvector(tfidf):
    result = []
    for row in tfidf:
        value = row[1]
        wdDict = dict.fromkeys(row[1].keys(), 0)
        for word, val in value.items():
            wdDict[word] = math.pow(val, 2)
        W = 0
        for word, val in wdDict.items():
            W+=float(val)
        akar = math.sqrt(W)
        result.append(akar)
    return result

@app.route('/test_tfidf')
def test_tfidf():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM testing WHERE typeir='tfidf' ORDER BY query ASC"
    cursor.execute(sql)
    result = cursor.fetchall()
    return render_template('test_tfidf.html', result=result)

@app.route('/test_doc2vec')
def test_doc2vec():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM testing WHERE typeir='doc2vec' ORDER BY query ASC"
    cursor.execute(sql)
    result = cursor.fetchall()
    return render_template('test_doc2vec.html', result=result)

@app.route('/test_nocluster')
def test_nocluster():
    conn = mysql.connect()
    cursor = conn.cursor()
    sql = "SELECT * FROM testing WHERE typeir='nocluster' ORDER BY query ASC"
    cursor.execute(sql)
    result = cursor.fetchall()
    return render_template('test_nocluster.html', result=result)