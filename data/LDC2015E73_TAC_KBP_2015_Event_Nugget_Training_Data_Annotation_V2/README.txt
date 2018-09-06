            TACKBP 2015 Event Nugget Training Data Annotation V2

                         LDC2015E73

                        August 17, 2015

                  Linguistic Data Consortium    
0.0 What's new
V2 of this package is updated to include event nugget XML files to
support Event Nugget Evaluation task 3: Event Nugget Coreference, 
in which performers are provided annotated event nuggets to identify
full event coreference links (event hoppers).


1. Introduction

Text Analysis Conference (TAC) is a series of workshops organized by
the National Institute of Standards and Technology (NIST). TAC was
developed to encourage research in natural language processing (NLP)
and related applications by providing a large test collection, common
evaluation procedures, and a forum for researchers to share their
results. Through its various evaluations, the Knowledge Base
Population (KBP) track of TAC encourages the development of systems
that can match entities mentioned in natural texts with those
appearing in a knowledge base and extract novel information about
entities from a document collection and add it to a new or existing
knowledge base.

Event Nugget Evaluation is a TAC KBP evaluation task which will be held 
in August, 2015. The goal of this task is to evaluate system performance 
on Event Nugget detection and Event Nugget coreference which requires 
systems to extract Event Nuggets and indicate the Event types, subtypes, 
the realis feature of the Event Nugget and coreference of Event Nuggets. 
For details about Event Nugget, Event types, subtypes and realis as well 
as Event coreference, please refer to the annotation guidelines under 
docs/. For this year's Event Nugget annotation, we follow the Event 
annotation guidelines in DEFT Rich ERE and adopt the concept of Event 
Hopper to tackle Event coreference. For detailed Event Nugget evaluation
specification, please refer to 

	http://www.nist.gov/tac/2015/KBP/Event/index.html 

and 

	http://cairo.lti.cs.cmu.edu/kbp/2015/event/index

This package contains 158 source documents and corresponding Event Nugget
annotation. The documents were originally annotated as eval data for the 
Event Nugget pilot Evaluation task in January, 2015. The original Event 
Nugget pilot evaluation annotation package was releases as LDC2015E69.  


2. Contents

./README.txt

  This file.

./docs
  DEFT_RICH_ERE_Annotation_Guidelines_English_Events_V2.9.pdf
	-- Event Nugget annotation guidelines
           (Event Arguments are not annotated)


  event_nugget_stats.tab
        -- File list, including token counts

  tackbp_event_nuggets.1.0.dtd -- DTD for event_nugget XML files
  tackbp_event_hoppers.1.0.dtd -- DTD for event_hopper XML files

./data
  The data directory holds the source and annotation files, 77 of which
  are discussion forum documents and 81 are newswire documents.
  
  The content is divided among two subdirectories:

./data/event_nugget/
  This directory holds all annotated event nugget annotation data in XML 
  format (in .xml)

./data/event_hopper/
  This directory holds all annotated event hopper and nugget annotation 
  data in XML format (in .xml)


./data/source/
  This directory holds all the source data (in .txt). 

3. Data Profile and Format

Genre	Files	Words	EventNugget	Hoppers 
-------------------------------------------------------------------------
NW	81	 27,897	 2,219		1,461		
DF	77	 97,124	 4,319		1,874
Total	158	125,021	 6,538		3,335
-------------------------------------------------------------------------

Event Nugget annotation files have a .event_nuggets.xml extension. Event 
Hopper annotation files have a .event_hoppers.xml extension, both are in 
XML format. 

For a full description of the elements, attributes, and structure of the 
annotation files, please see the DTDs in the docs directory of this release.

4. Using the Data

.1. Newswire Data

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

4.2 Multi-Post Discussion Forum Data

Multi-Post Discussion Forum files (CMP) are derived from English Discussion 
Forum threads.  They consist of a continuous run of posts from a thread but 
they are 800 words in length on average.  When taken from a short thread, a
CMP may comprise the entire thread.  However, when taken from longer threads,
it is a truncated version of its source, starting either with the preliminary 
post, or a post in the middle.

Some of the CMP files may contain characters in the range U+0085 - 
U+0099; these officially fall into a category of invisible "control" 
characters, but they all originated from single-byte "special 
punctuation" marks (quotes, etc. from CP1252) that were incorrectly 
transcoded to utf8.

The CMP txt files use the following markup framework, in which there 
may also be arbitrarily deep nesting of quote elements, and other 
elements may be present (e.g. "<a...>...</a>" anchor tags):

  <post ...>
  ...
  <quote ...>
  ...
  </quote>
  ...
  </post>
  ...

Note that CMP is an XML fragment rather than a full XML document and 
should be treated as plain text (not XML-parsed). 

4.3 Offset Calculation

All annotation XML files (file names "*.event_nuggets.xml") represent 
stand-off annotation of source files (file names "*.txt") and use offsets 
to refer to the text extents.

The event_mention XML elements all have attributes or contain sub-elements 
which use character offsets to identify text extents in the source.  
The offset gives the start character of the text extent; offset counting 
starts from the initial character, character 0, of the source document 
(.txt file) and includes newlines as well as all characters comprising 
XML-like tags in the source data.

4.3 Proper ingesting of XML

Because each source TXT document is extracted verbatim from source XML files,
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


5. Event Nugget Annotation Pipeline

5.1 Data selection

The documents annotated were part of Event Nugget Pilot Evaluation data set, 
which was released in LDC2015E69. 

5.2 Event Nugget Annotation

LDC modified previous Event Nugget annotation, following the Event annotation
guidelines for DEFT Rich ERE.  The previous Event Nugget annotation was performed
following the TAC-KBP-Event-Nugget-Detection-Annotation-Guidelines-v1.7. 
Annotators performed exhaustive Event Nugget and Event hopper annotation on 
each document, correcting Event Nugget annotation that was different as specified
in Rich ERE Event annotation guidelines, and adding Event Nugget and Event hopper 
annotation that was new in Rich ERE Event annotation. 

After annotation, corpus-wide quality checks (QC) were conducted by the team 
leader and senior annotators.  Refer to section 5.3 for detailed QC procedures.

Sometimes the discussion forum documents contain quoted texts either from an
external source or from the same document.  The quoted texts are annotated if 
they contain taggable Events. 

5.3 Quality Control

After manual quality control on individual files, LDC also conducted a
corpus-wide scan which included:

    -- Manual scan of Event triggers to review Event type and subtype values
    -- Scan all Event Nuggets to make sure annotation is complete
    -- Scan all Event Hoppers to make sure that Event mentions in the same 
       hoppers have same type and subtype value (except for mentions of contact 
       and transaction type, which only need to agree on type)

All identified outliers were then manually reviewed and corrected if needed. 

These manual QC checks were done in parallel with automatic validation
checks of the data during extraction and preparation of annotation files
for delivery.

6. Data Validation

For all text extent references, it was verified that the combination of docid, 
offset, and length was a valid reference to a string identical in content to
the XML text extent element.

 - Verified trigger text extent references valid
 - Verified each document in delivery received annotations

7. Know Issues
The event nugget IDs in both the event_hoppers.xml and event_nuggets.xml have 
not been anonymized and randomized. They are sorted according to their ids.
For evaluation, the event nugget IDs will be anomymized and randomized. System
development should not rely on the current event nugget IDs to predict event
coreference linking. 

Four Discussion Forum source files are slightly different from other DF files, 
namely they miss the XML header and hence can't be treated as valid XML files
when convert to xml format:

1b386c986f9d06fd0a0dda70c3b8ade9.txt
1ef1a80e902c1fc4507ca6285ab740ec.txt
3288ddfcb46d1934ad453afd8a4e292f.txt
3f71fead3fa119ccdcdf01769ffee5b1.txt

The eval data that you will receive during evaluation  will have the proper 
XML header and can be treated as valid XML files.

8. Copyright Information

(c) 2015 Trustees of the University of Pennsylvania

9. Contact Information

  Stephanie Strassel <strassel@ldc.upenn.edu>  PI
  Jonathan Wright    <jdwright@ldc.upenn.edu>  Technical oversight
  Zhiyi Song         <zhiyi@ldc.upenn.edu>     ERE annotation project manager
  Ann Bies           <bies@ldc.upenn.edu>      ERE annotation consultant
  Joe Ellis 	     <joellis@ldc.upenn.edu>   TAC KBP project manager

--------------------------------------------------------------------------
README created by Zhiyi Song on July 9, 2015
      modified by Dave Graff on July 9, 2015
      modified by Zhiyi Song on August 17, 2015

