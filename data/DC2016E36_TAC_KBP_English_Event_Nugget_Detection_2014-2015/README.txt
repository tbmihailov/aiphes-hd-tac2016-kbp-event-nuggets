			TAC KBP English Event Nugget Detection and Coreference
			Comprehensive Training and Evaluation Data 2014-2015
							    LDC2016E36

							  March 18, 2016
						Linguistic Data Consortium


1. Overview

This package contains training and evaluation data produced in support of
the TAC KBP English Event Nugget Detection and Coreference tasks in 2014
and 2015.

Text Analysis Conference (TAC) is a series of workshops organized by the
National Institute of Standards and Technology (NIST). TAC was developed to
encourage research in natural language processing (NLP) and related
applications by providing a large test collection, common evaluation
procedures, and a forum for researchers to share their results. Through its
various evaluations, the Knowledge Base Population (KBP) track of TAC
encourages the development of systems that can match entities mentioned in
natural texts with those appearing in a knowledge base and extract novel
information about entities from a document collection and add it to a new
or existing knowledge base.

The goal of the Event Nugget task is to evaluate system performance
on event nugget detection and event nugget coreference, which requires
systems to extract event nuggets, indicating for each the event type, 
subtype and realis feature, and then perform coreference of all event 
nuggets. More information about the Event Nugget evaluations can be 
found in the task descriptions included in this package as well as at
the Event Nugget section of NIST's 2015 TAC KBP website at
http://cairo.lti.cs.cmu.edu/kbp/2015/event/index

This package contains all evaluation and training data developed
in support of the TAC KBP Event Nugget evaluations during the two 
years in which the task was conducted, from 2014-2015. This includes 
gold standard event nugget annotations in multiple formats, coreference
information for the nuggets, and tokenization of the source documents,
as well as the source documents themselves.

The data included in this package were originally released to TAC KBP as:

LDC2014E121: DEFT Event Nugget Evaluation Training Data
LDC2015E03:  DEFT 2014 Event Nugget Evaluation Source Data
LDC2015E69:  DEFT 2014 Event Nugget Evaluation Annotation Data
LDC2015E73:  TAC KBP 2015 Event Nugget Training Data Annotation V2
LDC2015E94:  TAC KBP 2015 Event Nugget and Event Coreference Linking Evaluation Source Corpus
LDC2015R26:  TAC KBP 2015 Event Nugget and Event Corefence Linking

Summary of data included in this package:
+------+------------------+---------+
| Year | Source Documents | Nuggets |
+------+------------------+---------+
| 2014 |              351 |   10719 |
| 2015 |              360 |   12976 |
+------+------------------+---------+


2. Contents

./README.txt

  This file

./data/{2014|2015}/contents.txt

  The data in this package are organized by the year of original release
  in order to clarify dependencies, highlight occassional differences in
  formats from one year to another, and to increase readability in
  documentation. The contents.txt file within each year's root directory
  provides a list of the contents for all subdirectories as well as
  details about file formats and contents.

./docs/all_files.md5

  Paths (relative to the root of the corpus) and md5 checksums for all files
  in the package.

./docs/2014/TAC-KBP-Event-Nugget-Detection-Annotation-Guidelines-v1.7.pdf

  The guidelines used by annotators in 2014 in developing the gold 
  standard Event Nugget data contained in this corpus.

./docs/2014/Event_Nugget_Detection_Evaluation-v8.1.pdf

  Task Description for the 2014 Event Nugget evaluation track, written 
  by track coordinators.
  
./docs/2014/Event-Nugget-Detection-scoring-v17.pdf

  Scoring Description for the 2014 Event Nugget evaluation track, written 
  by track coordinators.
  
./docs/2015/DEFT_RICH_ERE_Annotation_Guidelines_English_Events_V2.9.pdf

  The guidelines used by annotators in 2015 in developing the gold 
  standard Event Nugget data contained in this corpus.

./docs/2015/Event_Mention_Detection_and_Coreference-2015-v1.1.pdf

  Task Description for the 2015 Event Nugget evaluation track, written 
  by track coordinators.
  
./docs/2015/Event-Mention-Detection-scoring-v27.pdf

  Scoring Description for the 2015 Event Nugget evaluation track, written 
  by track coordinators.

./dtd/tackbp_event_hoppers.1.0.dtd

  DTD for event_hopper XML files.
  
./dtd/tackbp_event_nuggets.1.0.dtd

  DTD for event_nugget XML files.
  
./tools/2014/*

  This directory contains the scripts and tools prepared by CMU for the
  2014 Event Nugget evaluation.  The file README.md included here describes
  the content in this directory and provides documentation for each
  script/tool.

./tools/2015/*

  This directory contains the scripts and tools prepared by CMU for the
  2015 Event Nugget evaluation.  The file README.md included here describes
  the content in this directory and provides documentation for each
  script/tool.


3. Annotation and Quality Control

2014 Event Nugget annotation adapted the event annotation guidelines from
LDC’s Light ERE annotation task with the additional annotation of event
attributes.

For the 2015 EN evaluation, event ‘triggers’ – the textual extent indicating
a reference to a valid event – was redefined as the smallest, contiguous 
extent of text that most saliently expresses the occurrence of an event.
Additionally, annotators for the 2015 data were allowed to ‘double tag’ event
triggers in order to indicate that a given text extent referred to more than one
event and was usually used to indicate the presence of an inferred event.
Event coreference was also added in 2015. Again taking from the Rich ERE 
task, EN addressed the challenge by adopting the notion of ‘event hoppers’, 
a more inclusive, less strict notion of event coreference as compared to 
previous approaches. Following this approach, event mentions are added to an 
event hopper when they “feel” coreferential to an annotator, even if they do 
not meet a strict event identity requirement. Event nuggets could be placed 
into the same event hoppers even if they differed in temporal or trigger 
granularity, their arguments were non-coreferential or conflicting, or even 
if their realis attribute differed.

Gold standard EN data was developed by first having two annotators perform event 
nugget annotation (which included the creation of event hoppers) independently 
for each document followed by an adjudication pass conducted by a senior 
annotator to resolve disagreements. The EN annotation team consisted of nine 
annotators, six of whom were also adjudicators and care was taken to ensure 
that annotators did not adjudicate their own first pass files. Following 
adjudication of all documents, a corpus-wide quality control pass was also 
performed.

Event Nugget and coreference annotation (2015) were performed simultaneously.
Annotators mark each event nugget - a string of text which specifically indicates
an event. After each Event Nugget is annotated, annotators need to make a decision
right away whether it is coreferenced with existing event hoppers.

During corpus wide quality control, annotators manually scan event triggers
to review event type and subtype values, ensure that generic event mentions are not
in the same hopper of other or actual event mentions, scan all event hoppers to make
sure that event mentions in the same hoppers have same type and subtype value (except
for mentions of contact and transaction type, which allows Contact.Contact and
Transaction.Transaction to be in the same hoppers of other contact subtypes and
transaction subtypes respectively) and manually review all identified outliers
and correct, if needed.

The annotation was then processed to serve different tracks of the event nugget
evaluation, using tool kits provided by evaluation coordinators:
Event Nugget annotation for Event Nugget Detection evaluation
Event Nugget and Event Hopper annotation for Event Nugget Detection and Coreference
Event Hopper annotation for Event Nugget Coreference


4. Source Documents

4.1 Newswire Data

Newswire data use the following markup framework:

  <DOC id="{doc_id_string}" type="{doc_type_label}">
  <HEADLINE>
  ...
  </HEADLINE>
  <DATELINE>
  ...
  </DATELINE>
  <TEXT>
  <P>
  ...
  </P>
  ...
  </TEXT>
  </DOC>

where the HEADLINE and DATELINE tags are optional (not always
present), and the TEXT content may or may not include "<P> ... </P>"
tags (depending on whether or not the "doc_type_label" is "story").

All the newswire files are parseable as XML but are treated as plain
text for annotation.

4.2 Multi-Post Discussion Forum Data

Multi-Post Discussion Forum files (MPDFs) are derived from English
Discussion Forum threads. They consist of a continuous run of posts
from a thread but they are only approximately 800 words in length
(excluding metadata and text within <quote> elements). When taken from
a short thread, a MPDF may comprise the entire thread. However, when
taken from longer threads, a MPDF is a truncated version of its
source, though it will always start with the preliminary post.

The MPDF files use the following markup framework, in which there may
also be arbitrarily deep nesting of quote elements, and other elements
may be present (e.g. "<a...>...</a>" anchor tags):

  <doc id="{doc_id_string}">
  <headline>
  ...
  </headline>
  <post ...>
  ...
  <quote ...>
  ...
  </quote>
  ...
  </post>
  ...
  </doc>

All the MPDF files are parseable as XML, but treated as plain text
in annotation.


5. Using the Data

5.1 Offset Calculation

All annotation XML files (file names "*.event_nuggets.xml") represent
stand-off annotation of source files (file names "*.txt") and use offsets
to refer to the text extents.

The event_mention XML elements all have attributes or contain sub-elements
which use character offsets to identify text extents in the source.
The offset gives the start character of the text extent; offset counting
starts from the initial character, character 0, of the source document
(.txt file) and includes newlines as well as all characters comprising
XML-like tags in the source data.

5.2 Proper ingesting of XML

Because each source text document is extracted verbatim from source XML files,
certain characters in its content (ampersands, angle brackets, etc.) are
escaped according to the XML specification.  The offsets of text extents
are based on treating this escaped text as-is (e.g. "&amp;" in a cmp.txt
file is counted as five characters).

Whenever any such string of "raw" text is included in a .Event_Nugget.xml
file (as the text extent to which an annotation is applied), a second
level of escaping has been applied, so that XML parsing of the XML
file will produce a string that exactly matches the source text.

For example, a reference to the corporation "AT&T" will appear in TXT as
"AT&amp;T".  Event Nugget annotation on this string would cite a length of 8
characters (not 4), and the string is stored in the XML file as
"AT&amp;amp;T" - when the XML file is parsed as intended, this will
return "AT&amp;T" to match the TXT content.


6. Acknowledgemnts

This material is based on research sponsored by Air Force Research
Laboratory and Defense Advance Research Projects Agency under
agreement number FA8750-13-2-0045. The U.S. Government is authoized
to reporoduce and distribute reprints for Governmental purposes
notwithstanding any copyright notation thereon. The views and
conclusions contained herein are those of the authors and should
not be interpreted as necessarily representing the official policies
or endorsements, either expressed or implied, of Air Force Research
Laboratory and Defense Advanced Research Projects Agency or the U.S.
Government.

The authors acknowledge the following contributors to this data set:
Dave Graff (LDC)
Xiaoyi Ma (LDC)
Justin Mott (LDC)
Tom Reise (LDC)
Hoa Dang (NIST)
Eduard Hovy (CMU)
Teruko Mitamura (CMU)
Boyan Onyshkevych (DARPA)


7. Copyright Information

(c) 2016 Trustees of the University of Pennsylvania


8. References

Joe Ellis, Jeremy Getman, Dana Fore, Neil Kuster, Zhiyi Song, Ann Bies, 
Stephanie Strassel. 2015. Overview of Linguistic Resources for the TAC KBP 
2015 Evaluations: Methodologies and Results. TAC KBP 2015 Workshop: National 
Institute of Standards and Technology, Gaithersburg, Maryland, November 16-17.


9. Contact Information

  Stephanie Strassel <strassel@ldc.upenn.edu>  PI
  Jonathan Wright    <jdwright@ldc.upenn.edu>  Technical oversight
  Zhiyi Song         <zhiyi@ldc.upenn.edu>     ERE annotation project manager
  Ann Bies           <bies@ldc.upenn.edu>      ERE annotation consultant
  Joe Ellis          <joellis@ldc.upenn.edu>   TAC KBP project manager
  Jeremy Getman      <jgetman@ldc.upenn.edu>   TAC KBP lead annotator

--------------------------------------------------------------------------
README created by Jeremy Getman on December 11, 2015
       updated by Jeremy Getman on December 14, 2015
       updated by Jeremy Getman on January 8, 2016
       updated by Jeremy Getman on January 12, 2016
	   updated by Dana Fore on February 5, 2016
	   updated by Dana Fore on March 15, 2016
