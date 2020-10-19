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
<h6 style="text-align:right">ID: {0}</h6>
\t<table>

        <caption style="text-align:left"><h6>{1}</h6></caption>
{2}
\t</table>

\t<table>

        <caption style="text-align:left"><h6>{3}</h6></caption>
{4}
\t</table>

insert.append("\t\t<tr>\n"
                              "\t\t\t<td class='name'>{0}</td>\n"
                              "\t\t\t<td class='des'>{1}</td>\n"
                              "\t\t</tr>\n"
                              .format(k, v))

\t<table>

        <caption style="text-align:left"><h6>{5}</h6></caption>
{6}
\t</table>

\t<table>

        <caption style="text-align:left"><h6>{7}</h6></caption>
{8}
\t</table>

\t<table>

        <caption style="text-align:left"><h6>{9}</h6></caption>
{10}
\t</table>
</body>
</html>

'''


css = '''
table {
  font-family: Roboto;
  width: 100%;margin-bottom: 20px;
    border: 2px solid #000000;
}

tr{
    border: 8px solid #000000;
}

td, th {
  border: 1px solid #dddddd;
  text-align: left;
  padding: 8px;
}

tr:nth-child(even) {
  background-color: #dddddd;
}

h6{
    padding: 0px;
    margin: 0px;
    font-size: 40px;
}

.name{
    font-size: 36px;
    width:30%;
}

.des {
    font-size: 36px;
    width:70%;
}'''

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


file_one = pd.read_excel("annot/gae-hom-hom.xlsx")
file_one = file_one.dropna()
file_one = file_one[['genesetID', 'Neighbour 1 Name', 'Neighbour 1 Desc',
                     'Neighbour 2 Name', 'Neighbour 2 Desc', 'Neighbour 3 Name',
                     'Neighbour 3 Desc', 'Neighbour 4 Name', 'Neighbour 4 Desc']]
file_one = file_one.set_index('genesetID').T.to_dict()


file_two = pd.read_excel("annot/gae-hom-onto.xlsx")
file_two = file_two.dropna()
file_two = file_two[['genesetID', 'Neighbour 1 Name', 'Neighbour 1 Desc',
                     'Neighbour 2 Name', 'Neighbour 2 Desc', 'Neighbour 3 Name',
                     'Neighbour 3 Desc', 'Neighbour 4 Name', 'Neighbour 4 Desc']]
file_two = file_two.set_index('genesetID').T.to_dict()


file_three = pd.read_excel("annot/gae-onto-onto.xlsx")
file_three = file_three.dropna()
file_three = file_three[['genesetID', 'Neighbour 1 Name', 'Neighbour 1 Desc',
                     'Neighbour 2 Name', 'Neighbour 2 Desc', 'Neighbour 3 Name',
                     'Neighbour 3 Desc', 'Neighbour 4 Name', 'Neighbour 4 Desc']]
file_three = file_three.set_index('genesetID').T.to_dict()


file_four = pd.read_excel("annot/jcd-hom-hom.xlsx")
file_four = file_four.dropna()
file_four = file_four[['genesetID', 'Neighbour 1 Name', 'Neighbour 1 Desc',
                     'Neighbour 2 Name', 'Neighbour 2 Desc', 'Neighbour 3 Name',
                     'Neighbour 3 Desc', 'Neighbour 4 Name', 'Neighbour 4 Desc']]
file_four = file_four.set_index('genesetID').T.to_dict()


file_five = pd.read_excel("annot/jcd-onto-onto.xlsx")
file_five = file_five.dropna()
file_five = file_five[['genesetID', 'Neighbour 1 Name', 'Neighbour 1 Desc',
                     'Neighbour 2 Name', 'Neighbour 2 Desc', 'Neighbour 3 Name',
                     'Neighbour 3 Desc', 'Neighbour 4 Name', 'Neighbour 4 Desc']]
file_five = file_five.set_index('genesetID').T.to_dict()


genesets = {}
for key in keys[0:1]:
    genesets[key] = [("A", getlist(file_one, key)),

                    ("B", getlist(file_two, key)),

                    ("C", getlist(file_three, key)),

                    ("D", getlist(file_four, key)),

                    ("E", getlist(file_five, key)),
                     ]


# Convert graphs into List of tuples
# genesets = {"50": [("A", [(0, 1), (3, 5)]), ("B", [(0, 1), (3, 5)]),
#          ("A", [("None", "None")]), ("B", [(0, 1), (3, 5)]),
#          ("A", [(0, 1), (3, 5)])]}

    for geneset in genesets:
        grphs = genesets[geneset]

        outer = []
        for i in grphs:
            insert = []
            for k, v in i[1]:
                insert.append("\t\t<tr>\n"
                              "\t\t\t<td class='name'>{0}</td>\n"
                              "\t\t\t<td class='des'>{1}</td>\n"
                              "\t\t</tr>\n"
                              .format(k, v))
            outer.append((i[0], insert))

        x = x.format(str(geneset), outer[0][0], ''.join(outer[0][1]),
                       outer[1][0], ''.join(outer[1][1]),
                       outer[2][0], ''.join(outer[2][1]),
                       outer[3][0], ''.join(outer[3][1]),
                       outer[4][0], ''.join(outer[4][1]))

        Html_file = open("survey/ind_tables/"+str(geneset)+".html", "w")
        Html_file.write(x)
        Html_file.close()

    HCTI_API_ENDPOINT = "https://hcti.io/v1/image"
    HCTI_API_USER_ID = '2b153f65-0a13-4000-9542-5235691e3da6'
    HCTI_API_KEY = '5e1bd040-b8cc-42ce-aebe-0fc97f7fef16'

    data = {'html': x,
            'css': css,
            'google_fonts': "Roboto"}

    image = requests.post(url=HCTI_API_ENDPOINT, data=data, auth=(HCTI_API_USER_ID, HCTI_API_KEY))

    print(geneset)
    print("Your image URL is: %s" % image.json()['url'])

    # input("Next")

    # https://hcti.io/v1/image/7ed741b8-f012-431e-8282-7eedb9910b32