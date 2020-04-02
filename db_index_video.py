import pickle
import numpy as np
import sqlite3 as sqlite
from os.path import basename


class Indexer(object):

    def __init__(self, db):
        self.con = sqlite.connect(db)

    def __del__(self):
        self.con.close()

    def db_commit(self):
        self.con.commit()

    def create_tables(self):
        self.con.execute('create table vidlist(filename)')
        self.con.execute('create index vid_idx on vidlist(filename)')

        self.con.execute('create table colorhists(vidid,features)')
        self.con.execute('create index colorhists_vidid_idx on colorhists(vidid)')

        self.con.execute('create table tempdiffs(vidid, features)')
        self.con.execute('create index tempdiffs_vidid_idx on tempdiffs(vidid)')

        self.con.execute('create table chdiffs(vidid, features)')
        self.con.execute('create index chdiffs_vidid_idx on chdiffs(vidid)')

        self.con.execute('create table blockdiffs(vidid, features)')
        self.con.execute('create index blockdiffs_vidid_idx on blockdiffs(vidid)')

    def add_to_index(self, vidname, descr):
        """ Take an video with feature descriptors,
            add to database. """

        print ('indexing', vidname)

        # get the imid
        vidid = self.get_id(vidname)

        # get features from descriptor
        colhist = descr['colhist']  # Nx3x256 np array
        chdiff = descr['chdiff'] # Nx3x256 np array
        tempdif = descr['tempdiff'] # Nx1 np array
        bdiff = descr['bdiff'] # Nx1 np array


        # store descriptor per video
        # use pickle to encode NumPy arrays as strings
        self.con.execute("insert into colorhists(vidid,features) values (?,?)", (vidid, pickle.dumps(colhist)))
        self.con.execute("insert into chdiffs(vidid,features) values (?,?)", (vidid,pickle.dumps(chdiff)))
        self.con.execute("insert into tempdiffs(vidid,features) values (?,?)", (vidid,pickle.dumps(tempdif)))
        self.con.execute("insert into blockdiffs(vidid,features) values (?,?)", (vidid,pickle.dumps(bdiff)))




    def get_id(self, vidname):
        """ Get an entry id and add if not present. """

        cur = self.con.execute("select rowid from vidlist where filename='%s'" % basename(vidname))
        res = cur.fetchone()
        if res == None:
            cur = self.con.execute("insert into vidlist(filename) values ('%s')" % basename(vidname))
            return cur.lastrowid
        else:
            return res[0]
