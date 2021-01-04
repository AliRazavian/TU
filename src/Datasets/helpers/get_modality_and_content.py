from yaml import dump as yamlDump
import pandas as pd

from Datasets.Modalities.implicit_number import Implicit_Number
from Datasets.Modalities.implicit_sequence import Implicit_Sequence
from Datasets.Modalities.implicit_plane import Implicit_Plane
from Datasets.Modalities.implicit_volume import Implicit_Volume
from Datasets.Modalities.style_number import Style_Number
from Datasets.Modalities.style_sequence import Style_Sequence
from Datasets.Modalities.style_plane import Style_Plane
from Datasets.Modalities.style_volume import Style_Volume
from Datasets.Modalities.id_from_indices import ID_from_Indices
from Datasets.Modalities.pseudo_label import Pseudo_Label
from Datasets.Modalities.one_vs_rest import One_vs_Rest
from Datasets.Modalities.bipolar import Bipolar
from Datasets.Modalities.multi_bipolar import Multi_Bipolar
from Datasets.Modalities.char_sequence import Char_Sequence
from Datasets.Modalities.image_from_filename import Image_from_Filename
from Datasets.Modalities.hierarchical_label import Hierarchical_Label
from Datasets.Modalities.multi_coordinate import Multi_Coordinate
from Datasets.helpers import Dictionary_Generator


def get_modality_and_content(
        annotations,
        modality_name: str,
        modality_cfgs: dict,
        ignore_index: int,
):
    content = None
    dg = Dictionary_Generator()
    if modality_cfgs['type'].lower() == 'Multi_Bipolar'.lower():
        assert 'columns' in modality_cfgs, f'modality_cfgs {modality_name}: {yamlDump} must contain "columns"'

        columns = Multi_Bipolar.get_csv_column_names(column_defintions=modality_cfgs['columns'],
                                                     modality_name=modality_name)
        ld = None
        if 'dictionary' in modality_cfgs:
            ld = modality_cfgs['dictionary']

        return Multi_Bipolar, annotations[columns], dg.get_bipolar_dictionary(
            modality_name=modality_name,
            label_dictionary=ld,
        )

    if modality_cfgs['type'].lower() == 'Multi_Coordinate'.lower():
        assert 'column_prefixes' in modality_cfgs
        columns = []
        for prefix in modality_cfgs['column_prefixes']:
            columns.append(f'{prefix}_x')
            columns.append(f'{prefix}_y')

        for colname in columns:
            assert colname in annotations.columns, f'The {colname} doesn\'t exist among columns in annotation'

        return Multi_Coordinate, annotations[columns], None

    dictionary = dg.get(modality_name)
    if modality_cfgs['type'].lower() == 'Implicit'.lower():
        assert('consistency' in modality_cfgs), \
            'modality_cfgs %s:%s must contain "consistency"' % (
                modality_name,
                yamlDump(modality_cfgs)
        )
        if modality_cfgs['consistency'].lower() == 'Number'.lower():
            return Implicit_Number, content, dictionary
        elif modality_cfgs['consistency'].lower() == '1D'.lower():
            return Implicit_Sequence, content, dictionary
        elif modality_cfgs['consistency'].lower() == '2D'.lower():
            return Implicit_Plane, content, dictionary
        elif modality_cfgs['consistency'].lower() == '3D'.lower():
            return Implicit_Volume, content, dictionary
        else:
            raise BaseException('Unknown consistency: "%s"' % (modality_cfgs['consistency']))

    elif modality_cfgs['type'].lower() == 'Style'.lower():
        if modality_cfgs['consistency'].lower() == 'Number'.lower():
            return Style_Number, content, dictionary
        elif modality_cfgs['consistency'].lower() == '1D'.lower():
            return Style_Sequence, content, dictionary
        elif modality_cfgs['consistency'].lower() == '2D'.lower():
            return Style_Plane, content, dictionary
        elif modality_cfgs['consistency'].lower() == '3D'.lower():
            return Style_Volume, content, dictionary

    elif modality_cfgs['type'].lower() == 'ID_from_Indices'.lower():
        return ID_from_Indices, pd.Series(ignore_index, index=annotations.index, dtype='int64'), dictionary
    elif modality_cfgs['type'].lower() == 'Pseudo_Label'.lower():
        return Pseudo_Label, content, dictionary

    else:
        assert('column_name' in modality_cfgs),\
            'modality_cfgs %s:\n%s\n must contain "column_name"' % (
                modality_name,
                yamlDump(modality_cfgs)
        )
        colname = modality_cfgs['column_name']
        if colname not in annotations:
            available_cols = ", ".join(str(i) for i in list(annotations))
            raise BaseException('Unknown column "%s" - in the dataset with cols: %s' % (colname, available_cols))

        content = annotations[colname]
        if modality_cfgs['type'].lower() == 'One_vs_Rest'.lower():
            return One_vs_Rest, content, dictionary
        elif modality_cfgs['type'].lower() == 'Bipolar'.lower():
            ld = None
            if 'dictionary' in modality_cfgs:
                ld = modality_cfgs['dictionary']

            return Bipolar, content, dg.get_bipolar_dictionary(
                modality_name=modality_name,
                label_dictionary=ld,
            )
        elif modality_cfgs['type'].lower() == 'Char_Sequence'.lower():
            return Char_Sequence, content, dictionary
        elif modality_cfgs['type'].lower() == 'Image_from_Filename'.lower():
            return Image_from_Filename, content, dictionary
        elif modality_cfgs['type'].lower() == 'Hierarchical_Label'.lower():
            return Hierarchical_Label, content, dictionary
        else:
            raise BaseException('Unknown column type: "%s"' % (modality_cfgs['type']))
