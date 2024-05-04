import pandas as pd
import requests
import numpy as np
import warnings
import copy
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")  # can remove the warning
import os
os.environ['NO_PROXY'] = 'nlm.nih.gov'



class MeshText:
    def __init__(self, mesh_list):
        # the mesh_list is a list of mesh_id
        self.mesh_list = mesh_list
        #self.disease_sim = self.get_Mesh_sim_mat(copy.deepcopy(self.mesh_list))


        self.ID2textid = {"D000925" : "68000925",
                          "D010975" : "68010975",
                          "D064907":"68064907",
                          "D000959":"68000959",
                          "D004232":"68004232",
                          "D014662":"68014662",
                          "D007166":"68007166",
                          "D017319":"68017319",
                          "D002492":"68002492",
                          "D006993":"68006993",
                          "D018490":"68018490",
                          "D000960":"68000960",
                          "D000700":"68000700",
                          "D016085":"68016085",
                          "D009466":"68009466",
                          "D014150":"68014150",
                          "D013566":"68013566",
                          "D001993":"68001993",
                          "D020533":"68020533",
                          "D000701":"68000701",
                          "D018663":"68018663"


                          }

    def get_text(self):
        text = []
        for disease in self.mesh_list:
            print("IDDDDDD:",disease)
            if disease in self.ID2textid.keys():
                disease = self.ID2textid[disease]
            #headers ={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.43'}
            api = f"https://www.ncbi.nlm.nih.gov/mesh/{disease}"
            #req = requests.get(api, headers=headers)
            req = requests.get(api)
            html = req.content.decode('utf-8', 'ignore')
            my_page = BeautifulSoup(html, 'lxml')

            for tag in my_page.find_all('div', class_='rprt abstract'):
                sub_tag = tag.find('p', class_="mesh_ds_scope_note")
                text.append(sub_tag.text)
        assert len(text)==len(self.mesh_list), "The text number is not much mesh_id number"
        return text



