"""This module gathers info of an MBS item from health.gov.au website."""

import re

import numpy as np
import pandas as pd
import requests

from bs4 import BeautifulSoup


class MBSOnline(object):
    def __init__(self, item):
        """This class can be used to interrogate `health.gov.au` to extract
        information of a given MBS item.

        Attributes that cannot be found on `health.gov.au` are set None as
        default.

        Attributes:
        --------------
        item: string
            MBS item code.

        url: string
            The base request URL.

        soup: bs4.BeautifulSoup
            Result of the HTML request.

        categories: list
            All the cagories in which the given MBS item belongs to.

        group: string
            Group to which the given MBS item belongs.

        subgroup: string
            Subgroup (if any) to which the given MBS item belongs.

        subheading: string
            Subheading to which the given MBS item belongs.

        description: string
            Textual description of the given MBS item.

        fee_start_date: # TODO

        description_start_date # TODO
        item_start_date # TODO

        fee: # TODO

        benefit75: # TODO

        benefit85: # TODO

        safety_net: # TODO
        """
        self.item = str(item)
        self.url = 'http://www9.health.gov.au/mbs/search.cfm?'
        self.soup = None
        self.category = None
        self.group = None
        self.subgroup = None
        self.subheading = None
        self.description = None
        self.fee_start_date = None
        self.description_start_date = None
        self.item_start_date = None
        self.fee = None
        self.benefit75 = None
        self.benefit85 = None
        self.safety_net = None

    def set_attributes(self):
        """asd."""
        self.soup = self.send_request(sopt='I')

        self.category = self.set_category()
        self.group, self.subgroup, self.subheading = self.set_info()
        self.description = self.set_description()
        dates = self.set_dates()
        if u'Schedule Fee Start Date:' in dates:
            self.fee_start_date = pd.to_datetime(dates[u'Schedule Fee Start Date:']).strftime('%Y-%m-%d')
        if u'Description Start Date:' in dates:
            self.description_start_date = pd.to_datetime(dates[u'Description Start Date:']).strftime('%Y-%m-%d')
        if 'Item Start Date:' in dates:
            self.item_start_date = pd.to_datetime(dates['Item Start Date:']).strftime('%Y-%m-%d')
        self.fee, self.benefit75, self.benefit85, self.safety_net = self.get_fees()
        return self

    def display(self):
        """Simply print the dataframe created with to_frame()."""
        print(self.to_frame())

    def to_frame(self):
        """Generate a pandas.DataFrame with the MBS info extracted."""
        df = pd.DataFrame(index=[self.item])
        df.loc[self.item, 'Category'] = self.category
        df.loc[self.item, 'Group'] = self.group
        df.loc[self.item, 'Subgroup'] = self.subgroup
        df.loc[self.item, 'Subheading'] = self.subheading
        df.loc[self.item, 'Description'] = self.description
        df.loc[self.item, 'Fee start date'] = self.fee_start_date
        df.loc[self.item, 'Description start date'] = self.description_start_date
        df.loc[self.item, 'Item start date'] = self.item_start_date
        df.loc[self.item, 'Fee (A$)'] = self.fee
        df.loc[self.item, 'Benefit 75% (A$)'] = self.benefit75
        df.loc[self.item, 'Benefit 85% (A$)'] = self.benefit85
        df.loc[self.item, 'Safety Net'] = self.safety_net
        return df.transpose()

    def send_request(self, sopt='I'):
        """Send request to `health.gov.au`.

        Parameters:
        --------------
        sopt: string
            Search option:
                - I: Search item numbers only
                - S: Search all notes and items (not implemented)

        Returns:
        --------------
        soup: bs4.BeautifulSoup
            HTML parsed object.
        """
        if sopt.upper() not in ['I', 'S']:
            raise ValueError('sopt can only be I or S')
        if sopt.upper() == 'S':
            raise NotImplementedError('Search all notes and items not yet '
                                      'implemented')
        r = requests.get(self.url, params={'q': self.item, 'sopt': sopt})
        return BeautifulSoup(r.text, 'html.parser')

    def set_category(self):
        """Extract the categories from the soup."""
        category_elem = filter(lambda x: 'category' in x.text.lower(),
                               self.soup.find_all('h3'))[0]
        return category_elem.text.split('<')[0]

    def set_info(self):
        """Extract group, subgroup and subheading from the soup."""
        group = None
        subgroup = None
        subheading = None
        out = []
        for elem in self.soup.find_all('div'):
            elem_class = elem.get('class')
            if (elem_class is not None) and (u'span9' in elem_class):
                out.append(elem.text)
        group = out[0]
        if len(out) > 1: subgroup = out[1]
        if len(out) > 2: subheading = out[2]
        return group, subgroup, subheading

    def set_description(self):
        """Extract the item description from the soup."""
        description = []
        for elem in self.soup.find_all('p'):
            if elem.get('align') == 'justify':
                description.append(elem.text.encode('utf-8').strip())
        return ' '.join(map(str, description))

    def set_dates(self):
        """Extract the relevant dates from the soup."""
        dates_keys = []
        dates_values = []
        for elem in self.soup.find_all('p'):
            if 'date' in elem.text.lower():
                for i, div_elem in enumerate(elem.find_all('div')):
                    if u'span8' in div_elem.get('class'):
                        dates_keys.append(div_elem.text)
                    elif u'span4' in div_elem.get('class'):
                        dates_values.append(div_elem.text)
        dates = {k: v for k, v in zip(dates_keys, dates_values)}
        return dates

    def get_fees(self):
        """Extract the fees from the soup."""
        fee = None  # FIXME is this redundant?
        benefit75 = None
        benefit85 = None
        safety_net = None
        for elem in self.soup.find_all('p'):
            for elem_p in elem.find_all('p'):
                if '$' in elem_p.text:
                    splitted = np.array(re.sub(' +', ' ', elem_p.text).split(' '))
                    if u'Fee:' in splitted:
                        fees = filter(lambda x: '$' in x, splitted)
                        fee = np.float(fees[0].split('$')[1])
                        benefit75 = np.float(fees[1].split('$')[1])
                        if len(fees) > 2:  # not always there
                            benefit85 = np.float(fees[2].split('$')[1])
                    elif u'Safety' in splitted:
                        safety_net = splitted[-1]
        return fee, benefit75, benefit85, safety_net

    def prettify(self):
        """Simple wrapper for BeautifulSoup prettify method."""
        if self.soup is not None:
            print(self.soup.prettify())
