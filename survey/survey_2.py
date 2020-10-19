import imgkit
import pandas as pd
import os
import pdfkit
import requests
import urllib.request

# String format of Html --- survey_design.html

x = '''<!DOCTYPE html>
<html>
<head><link rel="stylesheet" href="survey.css"></head>
<body>
\t<table>
        <caption style="text-align:left"><h3>{0}</h3></caption>
{1}
\t</table>
</body>
</html>
'''



def getlist(file_name, key):
    try:
        return [(file_name[key]['Neighbour 1 Name'], file_name[key]['Neighbour 1 Desc']),
                (file_name[key]['Neighbour 2 Name'], file_name[key]['Neighbour 2 Desc']),
                (file_name[key]['Neighbour 3 Name'], file_name[key]['Neighbour 3 Desc']),
                (file_name[key]['Neighbour 4 Name'], file_name[key]['Neighbour 4 Desc'])]
    except KeyError:
        return [("None", "None")]

keys = [271955.0, 34466.0, 127372.0, 614.0, 34620.0, 75542.0,
        34582.0, 127417.0, 617.0, 35882.0, 137565.0, 1214.0,
        46984.0, 14904.0, 35877.0, 135191.0, 34438.0, 75646.0,
        921.0, 75761.0, 137384.0, 135194.0, 271641.0, 75683.0,
        36285.0, 34682.0, 1766.0, 34302.0, 34549.0, 611.0, 34518.0,
        271950.0, 34560.0, 36299.0, 36287.0, 34513.0, 75623.0,
        34904.0, 34608.0, 14892.0, 34625.0, 137861.0, 271725.0,
        34552.0, 37188.0, 271940.0, 35883.0, 75657.0, 34970.0]


file_one = pd.read_excel("../annot/gae-hom-hom.xlsx")
file_one = file_one.dropna()
file_one = file_one[['genesetID', 'Neighbour 1 Name', 'Neighbour 1 Desc',
                     'Neighbour 2 Name', 'Neighbour 2 Desc', 'Neighbour 3 Name',
                     'Neighbour 3 Desc', 'Neighbour 4 Name', 'Neighbour 4 Desc']]
file_one = file_one.set_index('genesetID').T.to_dict()


file_two = pd.read_excel("../annot/gae-hom-onto.xlsx")
file_two = file_two.dropna()
file_two = file_two[['genesetID', 'Neighbour 1 Name', 'Neighbour 1 Desc',
                     'Neighbour 2 Name', 'Neighbour 2 Desc', 'Neighbour 3 Name',
                     'Neighbour 3 Desc', 'Neighbour 4 Name', 'Neighbour 4 Desc']]
file_two = file_two.set_index('genesetID').T.to_dict()


file_three = pd.read_excel("../annot/gae-onto-onto.xlsx")
file_three = file_three.dropna()
file_three = file_three[['genesetID', 'Neighbour 1 Name', 'Neighbour 1 Desc',
                     'Neighbour 2 Name', 'Neighbour 2 Desc', 'Neighbour 3 Name',
                     'Neighbour 3 Desc', 'Neighbour 4 Name', 'Neighbour 4 Desc']]
file_three = file_three.set_index('genesetID').T.to_dict()


file_four = pd.read_excel("../annot/jcd-hom-hom.xlsx")
file_four = file_four.dropna()
file_four = file_four[['genesetID', 'Neighbour 1 Name', 'Neighbour 1 Desc',
                     'Neighbour 2 Name', 'Neighbour 2 Desc', 'Neighbour 3 Name',
                     'Neighbour 3 Desc', 'Neighbour 4 Name', 'Neighbour 4 Desc']]
file_four = file_four.set_index('genesetID').T.to_dict()


file_five = pd.read_excel("../annot/jcd-onto-onto.xlsx")
file_five = file_five.dropna()
file_five = file_five[['genesetID', 'Neighbour 1 Name', 'Neighbour 1 Desc',
                     'Neighbour 2 Name', 'Neighbour 2 Desc', 'Neighbour 3 Name',
                     'Neighbour 3 Desc', 'Neighbour 4 Name', 'Neighbour 4 Desc']]
file_five = file_five.set_index('genesetID').T.to_dict()


genesets = {}
for key in keys:
    genesets[key] = [("A", getlist(file_one, key)),
                    ("B", getlist(file_two, key)),
                    ("C", getlist(file_three, key)),
                    ("D", getlist(file_four, key)),
                    ("E", getlist(file_five, key)),]

for geneset in genesets:
    grphs = genesets[geneset]
    for i in grphs:
        insert = []
        for j in i[1]:
            insert.append("\t\t<tr>\n"
                          "\t\t\t<td class='name'>{0}</td>\n"
                          "\t\t\t<td class='des'>{1}</td>\n"
                          "\t\t</tr>"
                          .format(str(j[0]), str(j[1])))
        html = x
        html = html.format(str(i[0]), ''.join(insert))
        Html_file = open("ind_tables/"+str(int(geneset))+"_"+i[0]+".html", "w", encoding='utf8')
        Html_file.write(html)
        Html_file.close()

