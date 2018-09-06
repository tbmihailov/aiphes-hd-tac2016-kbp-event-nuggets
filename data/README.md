# Data

The TAC KBP 2016 Event Nugget Detection data should be obtained from LDC:
DC2016E36_TAC_KBP_English_Event_Nugget_Detection_2014-2015
LDC2015E73_TAC_KBP_2015_Event_Nugget_Training_Data_Annotation_V2
LDC2016E71_TAC_KBP_2016_Eval_Core_Set_Rich_ERE_Annotation

The corresponding archive files should be extracted in the following structure:
```

.
├── DC2016E36_TAC_KBP_English_Event_Nugget_Detection_2014-2015
│   ├── data
│   │   ├── 2014
│   │   │   ├── eval
│   │   │   │   ├── annotation
│   │   │   │   ├── source
│   │   │   │   └── token_offset
│   │   │   └── training
│   │   │       ├── annotation
│   │   │       ├── source
│   │   │       └── token_offset
│   │   └── 2015
│   │       ├── eval
│   │       │   ├── bratHopperAnn
│   │       │   ├── bratNuggetAnn
│   │       │   ├── hopper
│   │       │   ├── nugget
│   │       │   ├── source
│   │       │   ├── tbf
│   │       │   └── tkn
│   │       └── training
│   │           ├── bratHopperAnn
│   │           ├── bratNuggetAnn
│   │           ├── event_hopper
│   │           ├── event_nugget
│   │           └── source
│   ├── docs
│   │   ├── 2014
│   │   └── 2015
│   ├── dtd
│   └── tools
│       ├── 2014
│       │   ├── LICENSE
│       │   └── example_data
│       │       ├── ann
│       │       ├── tkn
│       │       └── txt
│       └── 2015
│           ├── LICENSE
│           ├── bin
│           ├── data
│           │   ├── conversion_demo
│           │   │   ├── ann
│           │   │   ├── ere
│           │   │   └── tkn
│           │   ├── scoring_demo
│           │   │   ├── ann
│           │   │   ├── tkn
│           │   │   └── txt
│           │   └── test_cases
│           │       ├── conll_tests
│           │       │   ├── ann
│           │       │   ├── old_conll
│           │       │   ├── tkn
│           │       │   └── txt
│           │       ├── mention_detection_tests
│           │       └── wrong_format_tests
│           │           └── tkn
│           ├── doc
│           ├── ldc-xml-to-brat-converter
│           │   ├── data
│           │   ├── doc
│           │   └── src
│           │       └── main
│           │           ├── java
│           │           │   ├── converter
│           │           │   └── net
│           │           │       └── junaraki
│           │           │           └── annobase
│           │           │               ├── io
│           │           │               ├── pipeline
│           │           │               ├── process
│           │           │               ├── type
│           │           │               │   ├── concept
│           │           │               │   ├── element
│           │           │               │   ├── graph
│           │           │               │   └── relation
│           │           │               │       ├── concept
│           │           │               │       └── element
│           │           │               └── util
│           │           │                   └── graph
│           │           └── resources
│           ├── reference-coreference-scorers-8.01
│           │   └── test
│           │       └── DataFiles
│           ├── token-file-maker
│           │   ├── data
│           │   │   └── input
│           │   └── src
│           │       └── main
│           │           └── java
│           │               └── evmeval
│           ├── util
│           ├── visualization
│           │   ├── css
│           │   │   └── static
│           │   │       └── fonts
│           │   └── js
│           │       ├── client
│           │       │   └── src
│           │       └── static
│           │           └── fonts
│           └── visualization_tools
├── LDC2015E73_TAC_KBP_2015_Event_Nugget_Training_Data_Annotation_V2
│   ├── data
│   │   ├── event_hopper
│   │   ├── event_nugget
│   │   └── source
│   └── docs
├── LDC2016E71_TAC_KBP_2016_Eval_Core_Set_Rich_ERE_Annotation
│   └── data
│       └── cmn
│           └── df
│               ├── ere
│               └── source
└── clear_data

```

## Prepare the data

For converting the training data:
```bash
# Set the directories to the downloaded data in data_conversion.sh
bash data/data_conversion.sh

```


For converting the evaluation data:

```bash
# Set the directories to the downloaded data in data_conversion_2016eval.sh
bash data/data_conversion_2016eval.sh

```
