%% acmsmall.cls - version 1.9  (with ACM Reference Format)
%% Aptara Inc., dated 9 July 2012 (V1.4)
%% Boris Veysman, 09 August 2013
%% (c) 2011-2013 Association for Computing Machinery (ACM)
%% For small trim journals
%%
%% Based on ESUB2ACM.CLS V1.2 - November 10th. 1999
%%
%% If you face any problem while working with this class file or have any feedback/suggestion, 
%% please contact ACM Support helpline at: "acmtexsupport@aptaracorp.com".
%% Users can also go through the FAQs available on the journal's
%% submission webpage.
%%
%% Version 1.10
%% Put copyright code into a .sty 
%%
%%
%% Version 1.9
%% New code for copyright
%% 
%% Version 1.8
%% New Copyright statements
%%
%%
%% Version 1.7
%% Added support for CCSXML
%%
%% Version 1.6
%% -----------
%% Redefined concepts
%%
%%
%% Version 1.5
%% -----------
%% New command \ccsdesc[significance]{Concept~Concept}
%%
%% Version 1.4 (History)
%% ---------------------
%% 1) All citation formats are also redefined after
%%    checking the loading condition of natbib.sty 
%% 2) Added an option called 'prodmode', which uses New Century School Book
%%    and Helvetica as base and second font respectively after
%%    feedback from Joanne (Dated 28/06/2010) to simulate print output.
%% 3) Changed running head style (Joanne 02/07/2010)
%% 4) Added three new transactions (Jono 14/07/2010)
%% 5) Added four new transactions/journals (Joanne 04/11/2011)
%% 6) Added newtheorem definition for 'conjecture'
%% 7) Trimmed the 'double outputting' of the DOI/URL beneath ACM Reference Format (before 1. INTRODUCTION)
%%    and also beneath the Permission Statement/copyright line. Also inserted the "http://dx.doi.org" stem. (Gerry Murray, March 2012)
%% 8) Changed the price charged/article information from $10.00 to $15.00 (Gerry 24/05/2012)
%% 9) Added 'TEAC' - July 9th. 2012
%% 10) Added 'TOPC' - January 18th. 2013 
%% 11) Added TOMM from acmlarge design on 2014/05/28 (Aptara)
%% 12) Added new journals (TSAS and TOMPECS) on 2014/09/10 (Aptara)
%% 13) a. Change the journal name TALLIP (formerly TALIP) 2014/12/09 (Aptara)
%%     b. Added issn numbers for TOMPECS 2014/12/09 (Aptara)
%% 14) Added new journal TCPS on 2015/08/27 (Aptara)
%% 15) Added new journal TSC on 2016/03/02 (Aptara)
%% 16) Journal TISSEC is being renamed to TOPS on 2016/04/20 (Aptara)
%%
%% Steps to compile: latex, bibtex, latex latex
%%
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}
%%
%% Bibliographic cite forms needed:
%%
%%  \cite{key}
%%    which produces citations with author list and year.
%%    eg. [Brown 1978; Jarke, et al. 1985]
%%  \citeA{key}
%%    which produces citations with only the author list.
%%    eg. [Brown; Jarke, et al.]
%%  \citeN{key}
%%    which produces citations with the author list and year, but
%%    can be used as nouns in a sentence; no brackets appear around
%%    the author names, but only around the year.
%%      eg. Shneiderman [1978] states that......
%%    \citeN should only be used for a single citation.
%%    \citeNN{refkey1,refkey2} for author [ref1year; ref2year]
%%    \citeyear{key}
%%        which produces the year information only, within brackets.
%%
%% Abbreviated author lists use the ``et al.'' construct.
%%
%% The above are examples of required ACM bibliographic cite formats needed.
%% *******************
%% Here is the complete list of cite forms from the chicago bibliographic style
%%
%%  \cite{key}
%%    which produces citations with abbreviated author list and year.
%%  \citeNP{key}
%%    which produces citations with abbreviated author list and year.
%%  \citeA{key}
%%    which produces only the abbreviated author list.
%%  \citeANP{key}
%%    which produces only the abbreviated author list.
%%  \citeN{key}
%%    which produces the abbreviated author list and year, with only the
%%    year in parentheses. Use with only one citation.
%%  \citeyear{key}
%%    which produces the year information only, within parentheses.
%%  \citeyearNP{key}
%%    which produces the year information only.
%%
%% Abbreviated author lists use the ``et al.'' construct.
%%
%% `NP' means `no parentheses' 
%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{acmsmall}[2013/08/09 v1.10 ACM class]

\def\fileversion{v1.10}            
\def\filedate{August 10, 2013}

% Metadata Information
\def\@acmVolume{V} %the volume
\def\@acmNumber{N} %the number
\def\@acmArticle{A} %article number
\def\@articleSeq{1} %article Sequence
\def\@acmPrice{15.00} %article price  % Changed to 15 - June 2012 - Gerry
\def\@acmYear{YYYY} %the last two digits of the year, 
\def\@acmMonth{1}  %the month
\def\@journalName{ACM Journal Name} %the name of the ACM journal
\def\@journalNameShort{ACM} %the acronym of the ACM journal 
\def\@journalCode{jn} %the code of the ACM journal 
\def\@permissionCodeOne{0000-0000} %the permission code of the ACM journal
\def\@doi{http://dx.doi.org/10.1145/0000000.0000000} % These 'default' '0' values are over-ridden, during production, with 'correct' numbers entered via source .tex file - Gerry March 2012

% ----------------
% Gerry - April 2011 - To assist in the formatting of the NEW ACM Reference format - 'DOI' in tt font, and url string in default
\usepackage{url}
\def\UrlFont{}
\newcommand{\BIBand}{\&}
\def\showDOI#1{\mbox{{\tt DOI:}{#1}}} % Gerry April 2011 - forces the DOI: to 'stay' with the 'url-(string)link' (no need for space at end either)
% ----------------

\if@compatibility\else
\DeclareOption{letterpaper}
   {\setlength\paperheight {11in}%
    \setlength\paperwidth  {8.5in}%
    \setlength\voffset     {-38.2pt}%
    \setlength\hoffset     {-8.25pt}%
    \def\special@paper{8.5in,11in}
	% Needed to set PDF page size
	\special{papersize=8.5in,11in}}
\DeclareOption{landscape}
   {\setlength\@tempdima   {\paperheight}%
    \setlength\paperheight {\paperwidth}%
    \setlength\paperwidth  {\@tempdima}}
\fi

\DeclareOption{oneside}{\@twosidefalse \@mparswitchfalse}
\DeclareOption{twoside}{\@twosidetrue  \@mparswitchtrue}

\DeclareOption{final}{\setlength\overfullrule{0pt}}

\DeclareOption{10pt}{\def\@ptsize{0}} %needed for amssymbols.sty 
\DeclareOption{11pt}{\ClassError{acmsmall}{11pt style not supported}
                        {ACM transactions/journals documents can be set in 10pt only}}
\DeclareOption{12pt}{\ClassError{acmtrans}{12pt style not supported}
                        {ACM transactions/journals documents can be set in 10pt only}}

\newif\ifprod@mode\prod@modefalse
\DeclareOption{prodmode}{\global\prod@modetrue
\typeout{ACM, Production Font style: 2010/06/28 by Aptara}}

\newif\if@acmtopc
\newif\if@acmjacm
\newif\if@acmtops
\newif\if@acmtocl
\newif\if@acmtocs
\newif\if@acmtochi
\newif\if@acmtodaes
\newif\if@acmtods
\newif\if@acmtois
\newif\if@acmtomacs
\newif\if@acmtoms
\newif\if@acmtoplas
\newif\if@acmtosem
\newif\if@acmtoit
\newif\if@acmtecs
\newif\if@acmtallip
\newif\if@acmjeric
\newif\if@acmtaco
\newif\if@acmjea
\newif\if@acmtslp
\newif\if@acmcie
\newif\if@acmtos
\newif\if@acmcsur
\newif\if@acmjetc
\newif\if@acmtosn
\newif\if@acmtalg
\newif\if@acmtaas
\newif\if@acmtweb
\newif\if@acmtkdd
\newif\if@acmtrets
\newif\if@acmtmis
\newif\if@acmtiis
\newif\if@acmtist
\newif\if@acmtoct
\newif\if@acmjdiq
\newif\if@acmtaccess
\newif\if@acmtoce
\newif\if@acmteac
\newif\if@acmtomm
\newif\if@acmtsas
\newif\if@acmtompecs
\newif\if@acmtcps
\newif\if@acmtsc

% Journal Options
\DeclareOption{acmtopc}{
  \typeout{}
  \typeout{Using ACM, TOPC's option: 2013/01/18 by Aptara}
  \typeout{}
  \global\@acmtopctrue
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Parallel Computing}
  \def\@journalNameShort{ACM Trans. Parallel Comput.}
  \def\@journalCode{topc}        
  \def\@permissionCodeOne{1539-9087} 
}

\DeclareOption{acmjacm}{
  \typeout{}
  \typeout{Using ACM, JACM's option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmtrue
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{Journal of the ACM}
  \def\@journalNameShort{J. ACM}        
  \def\@journalCode{jacm}        
  \def\@permissionCodeOne{0004-5411} 
}

\DeclareOption{acmtops}{
  \typeout{}
%  \typeout{Using ACM, TISSEC's option: 2010/05/04 by Aptara}
  \typeout{Using ACM, TOPS's option: 2016/04/20 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopstrue
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
%  \def\@journalName{ACM Transactions on Information and System Security}
%  \def\@journalNameShort{ACM Trans. Info. Syst. Sec.}
  \def\@journalName{ACM Transactions on Privacy and Security}
  \def\@journalNameShort{ACM Trans. Priv. Secur.}
  \def\@journalCode{tops}
%  \def\@permissionCodeOne{1094-9224}
  \def\@permissionCodeOne{2471-2566}
}

\DeclareOption{acmtocl}{
  \typeout{}
  \typeout{Using ACM, TOCL's option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtocltrue
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Computational Logic}
  \def\@journalNameShort{ACM Trans. Comput. Logic}
  \def\@journalCode{tocl}        
  \def\@permissionCodeOne{1529-3785} 
}

\DeclareOption{acmtocs}{
  \typeout{}
  \typeout{Using ACM, TOCS's option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocstrue
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Computer Systems}
  \def\@journalNameShort{ACM Trans. Comput. Syst.}
  \def\@journalCode{tocs}        
  \def\@permissionCodeOne{0734-2071} 
}

\DeclareOption{acmtochi}{
  \typeout{}
  \typeout{Using ACM, TOCHI's option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochitrue
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Computer-Human Interaction}
  \def\@journalNameShort{ACM Trans. Comput.-Hum. Interact.}
  \def\@journalCode{tochi}        
  \def\@permissionCodeOne{1073-0516} 
}

\DeclareOption{acmtodaes}{
  \typeout{}
  \typeout{Using ACM, TODAES option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaestrue
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Design Automation of Electronic Systems}
  \def\@journalNameShort{ACM Trans. Des. Autom. Electron. Syst.}
  \def\@journalCode{todaes}        
  \def\@permissionCodeOne{1084-4309} 
}

\DeclareOption{acmtods}{
  \typeout{}
  \typeout{Using ACM, TODS's option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodstrue
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Database Systems}
  \def\@journalNameShort{ACM Trans. Datab. Syst.}
  \def\@journalCode{tods}        
  \def\@permissionCodeOne{0362-5915}
}

\DeclareOption{acmtois}{
  \typeout{}
  \typeout{Using ACM, TOIS's option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoistrue
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Information Systems}
  \def\@journalNameShort{ACM Trans. Inf. Syst.}
  \def\@journalCode{tois}        
  \def\@permissionCodeOne{1046-8188} 
}

\DeclareOption{acmtomacs}{
  \typeout{}
  \typeout{Using ACM, TOMACS's option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacstrue
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Modeling and Computer Simulation}
  \def\@journalNameShort{ACM Trans. Model. Comput. Simul.}
  \def\@journalCode{tomacs}
  \def\@permissionCodeOne{1049-3301} 
}

\DeclareOption{acmtoms}{
  \typeout{}
  \typeout{Using ACM, TOMS's option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomstrue
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Mathematical Software}
  \def\@journalNameShort{ACM Trans. Math. Softw.}
  \def\@journalCode{toms}        
  \def\@permissionCodeOne{0098-3500} 
}

\DeclareOption{acmtoplas}{
  \typeout{}
  \typeout{Using ACM, TOPLAS option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplastrue
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Programming Languages and Systems}
  \def\@journalNameShort{ACM Trans. Program. Lang. Syst.}
  \def\@journalCode{toplas}        
  \def\@permissionCodeOne{0164-0925} 
}

\DeclareOption{acmtosem}{
  \typeout{}
  \typeout{Using ACM, TOSEM option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemtrue
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Software Engineering and Methodology}
  \def\@journalNameShort{ACM Trans. Softw. Eng. Methodol.}
  \def\@journalCode{tosem}        
  \def\@permissionCodeOne{1049-331X} 
}

\DeclareOption{acmtoit}{
  \typeout{}
  \typeout{Using ACM, TOIT option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoittrue
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Internet Technology}
  \def\@journalNameShort{ACM Trans. Internet Technol.}
  \def\@journalCode{toit}        
  \def\@permissionCodeOne{1533-5399} 
}

\DeclareOption{acmtecs}{
  \typeout{}
  \typeout{Using ACM, TECS option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecstrue
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Embedded Computing Systems}
  \def\@journalNameShort{ACM Trans. Embedd. Comput. Syst.}
  \def\@journalCode{tecs}
  \def\@permissionCodeOne{1539-9087} 
}

%% Updaded Talip for change in the journal name to TALLIP (9th Dec. 2014)
\DeclareOption{acmtallip}{
  \typeout{}
%  \typeout{Using ACM, TALIP option: 2010/05/04 by Aptara}
  \typeout{Using ACM, TALLIP option: 2014/12/09 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtalliptrue
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
%  \def\@journalName{ACM Transactions on Asian Language Information Processing}
%  \def\@journalNameShort{ACM Trans. Asian Lang. Inform. Process.}
  \def\@journalName{ACM Transactions on Asian and Low-Resource Language Information Processing}
  \def\@journalName{ACM Trans. Asian Low-Resour. Lang. Inf. Process.}
  \def\@journalNameShort{ACM Trans. Asian Low-Resour. Lang. Inf. Process.}
  \def\@journalCode{tallip}        
%  \def\@permissionCodeOne{1530-0226} 
  \def\@permissionCodeOne{2375-4699}
}

\DeclareOption{acmjeric}{
  \typeout{}
  \typeout{Using ACM, JERIC option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjerictrue
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Journal of Educational Resources in Computing}
  \def\@journalNameShort{ACM J. Edu. Resources in Comput.}
  \def\@journalCode{jeric}        
  \def\@permissionCodeOne{1073-0516} 
}

\DeclareOption{acmtaco}{
  \typeout{}
  \typeout{Using ACM, TACO option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacotrue
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Architecture and Code Optimization}
  \def\@journalNameShort{ACM Trans. Architec. Code Optim.}
  \def\@journalCode{taco}        
  \def\@permissionCodeOne{1544-3566} 
}

\DeclareOption{acmjea}{
  \typeout{}
  \typeout{Using ACM, JEA option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeatrue
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Journal of Experimental Algorithmics}
  \def\@journalNameShort{ACM J. Exp. Algor.}
  \def\@journalCode{jea}        
  \def\@permissionCodeOne{1084-6654} 
}

\DeclareOption{acmtslp}{
  \typeout{}
  \typeout{Using ACM, TSLP option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslptrue
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Speech and Language Processing}
  \def\@journalNameShort{ACM Trans. Speech Lang. Process.}
  \def\@journalCode{tslp}        
  \def\@permissionCodeOne{1550-4875} 
}

\DeclareOption{acmcie}{
  \typeout{}
  \typeout{Using ACM, CIE option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmcietrue
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Computers in Entertainment}
  \def\@journalNameShort{ACM Comput. Entertain.}
  \def\@journalCode{cie}        
  \def\@permissionCodeOne{1544-3574} 
}

\DeclareOption{acmtos}{
  \typeout{}
  \typeout{Using ACM, TOS option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtostrue
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Storage}
  \def\@journalNameShort{ACM Trans. Storage}
  \def\@journalCode{tos}        
  \def\@permissionCodeOne{1553-3077} 
}

\DeclareOption{acmcsur}{
  \typeout{}
  \typeout{Using ACM, CSUR option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurtrue
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Computing Surveys}
  \def\@journalNameShort{ACM Comput. Surv.}
  \def\@journalCode{csur}        
  \def\@permissionCodeOne{0360-0300} 
}

\DeclareOption{acmjetc}{
  \typeout{}
  \typeout{Using ACM, JETC option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetctrue
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Journal on Emerging Technologies in Computing Systems}
  \def\@journalNameShort{ACM J. Emerg. Technol. Comput. Syst.}
  \def\@journalCode{jetc}
  \def\@permissionCodeOne{1550-4832} 
}

\DeclareOption{acmtosn}{
  \typeout{}
  \typeout{Using ACM, TOSN option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosntrue
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Sensor Networks}
  \def\@journalNameShort{ACM Trans. Sensor Netw.}
  \def\@journalCode{tosn}
  \def\@permissionCodeOne{1550-4859} 
}

\DeclareOption{acmtalg}{
  \typeout{}
  \typeout{Using ACM, TALG option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgtrue
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Algorithms}
  \def\@journalNameShort{ACM Trans. Algor.}
  \def\@journalCode{talg}
  \def\@permissionCodeOne{1549-6325} 
}

\DeclareOption{acmtaas}{
  \typeout{}
  \typeout{Using ACM, TAAS option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaastrue
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Autonomous and Adaptive Systems}
  \def\@journalNameShort{ACM Trans. Autonom. Adapt. Syst.}
  \def\@journalCode{taas}
  \def\@permissionCodeOne{1556-4665} 
}

\DeclareOption{acmtweb}{
  \typeout{}
  \typeout{Using ACM, TWEB option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebtrue
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on the Web}
  \def\@journalNameShort{ACM Trans. Web}
  \def\@journalCode{tweb}
  \def\@permissionCodeOne{1559-1131} 
}

\DeclareOption{acmtkdd}{
  \typeout{}
  \typeout{Using ACM, TKDD option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddtrue
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Knowledge Discovery from Data}
  \def\@journalNameShort{ACM Trans. Knowl. Discov. Data.}
  \def\@journalCode{tkdd}
  \def\@permissionCodeOne{1556-4681} 
}

\DeclareOption{acmtrets}{
  \typeout{}
  \typeout{Using ACM, TRETS option: 2010/05/04 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretstrue
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Reconfigurable Technology and Systems}
  \def\@journalNameShort{ACM Trans. Reconfig. Technol. Syst.}
  \def\@journalCode{trets}
  \def\@permissionCodeOne{1936-7406} 
}

\DeclareOption{acmtmis}{
  \typeout{}
  \typeout{Using ACM, TMIS option: 2010/07/14 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmistrue
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Management Information Systems}
  \def\@journalNameShort{ACM Trans. Manag. Inform. Syst.}
  \def\@journalCode{tmis}
  \def\@permissionCodeOne{2158-656X} 
}

\DeclareOption{acmtiis}{
  \typeout{}
  \typeout{Using ACM, TIIS option: 2010/07/14 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiistrue
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Interactive Intelligent Systems}
  \def\@journalNameShort{ACM Trans. Interact. Intell. Syst.}
  \def\@journalCode{tiis}
  \def\@permissionCodeOne{2160-6455} 
}

\DeclareOption{acmtist}{
  \typeout{}
  \typeout{Using ACM, TIST option: 2010/07/14 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtisttrue
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Intelligent Systems and Technology}
  \def\@journalNameShort{ACM Trans. Intell. Syst. Technol.}
  \def\@journalCode{tist}
  \def\@permissionCodeOne{2157-6904} 
}

\DeclareOption{acmtoct}{
  \typeout{}
  \typeout{Using ACM, TOCT option: 2011/10/31 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtocttrue
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Computation Theory}
  \def\@journalNameShort{ACM Trans. Comput. Theory}
  \def\@journalCode{toct}
  \def\@permissionCodeOne{1942-3454} 
}

\DeclareOption{acmjdiq}{
  \typeout{}
  \typeout{Using ACM, JDIQ option: 2011/10/31 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqtrue
  \global\@acmtaccessfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Journal of Data and Information quality}
  \def\@journalNameShort{ACM J. Data Inform. Quality}
  \def\@journalCode{jdiq}
  \def\@permissionCodeOne{1936-1955} 
}

\DeclareOption{acmtaccess}{
  \typeout{}
  \typeout{Using ACM, TACCESS option: 2011/10/31 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccesstrue
  \global\@acmtallipfalse
  \global\@acmtiisfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Accessible Computing}
  \def\@journalNameShort{ACM Trans. Access. Comput.}
  \def\@journalCode{taccess}
  \def\@permissionCodeOne{1936-7228} 
}

\DeclareOption{acmtoce}{
  \typeout{}
  \typeout{Using ACM, TOCE option: 2011/10/31 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtallipfalse
  \global\@acmtiisfalse
  \global\@acmtocetrue
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Computing Education}
  \def\@journalNameShort{ACM Trans. Comput. Educ.}
  \def\@journalCode{toce}
  \def\@permissionCodeOne{1946-6226} 
}

\DeclareOption{acmteac}{
  \typeout{}
  \typeout{Using ACM, TEAC option: 2012/07/04 by Orna Agmon Ben-Yehuda}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtallipfalse
  \global\@acmtiisfalse
  \global\@acmtocefalse
  \global\@acmteactrue
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Economics and Computation}
  \def\@journalNameShort{ACM Trans. Econ. Comp.}
  \def\@journalCode{teac}
  \def\@permissionCodeOne{1946-6227} 
}
%
%%
%% TeXSupport: Added TOMM from acmlarge design on 2014/05/28
%%
\DeclareOption{acmtomm}{
  \typeout{}
  \typeout{Using ACM, TOMM's option: 2014/05/28 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtallipfalse
  \global\@acmtiisfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommtrue
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
%  \def\@journalName{ACM Transactions on Multimedia Computing, Communications and Applications}
  \def\@journalName{ACM Trans. Multimedia Comput. Commun. Appl.}        
  \def\@journalNameShort{ACM Trans. Multimedia Comput. Commun. Appl.}        
  \def\@permissionCodeOne{1551-6857} 
  \def\@permissionCodeTwo{0100} 
}
%
%%
%% TeXSupport: Added new journal TSAS on 2014/09/10
%%
\DeclareOption{acmtsas}{
  \typeout{}
  \typeout{Using ACM, TSAS's option: 2014/09/10 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtallipfalse
  \global\@acmtiisfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsastrue
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Spatial Algorithms and Systems}
  \def\@journalNameShort{ACM Trans. Spatial Algorithms Syst.}        
  \def\@permissionCodeOne{2374-0353} 
}
%
%%
%% TeXSupport: Added new journal TOMPECS on 2014/09/10
%%
\DeclareOption{acmtompecs}{
  \typeout{}
  \typeout{Using ACM, TOMPECS's option: 2014/09/10 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtallipfalse
  \global\@acmtiisfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecstrue
  \global\@acmtcpsfalse
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Modeling and Performance Evaluation of Computing Systems}
  \def\@journalName{ACM Trans. Model. Perform. Eval. Comput. Syst.}        
  \def\@journalNameShort{ACM Trans. Model. Perform. Eval. Comput. Syst.}        
  \def\@permissionCodeOne{2376-3639}
}
%
%%
%% TeXSupport: Added new journal TCPS on 2015/08/27
%%
\DeclareOption{acmtcps}{
  \typeout{}
  \typeout{Using ACM, TCPS's option: 2015/08/27 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtallipfalse
  \global\@acmtiisfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpstrue
  \global\@acmtscfalse
  \def\@journalName{ACM Transactions on Cyber-Physical Systems}
  \def\@journalNameShort{ACM Trans. Cyber-Phys. Syst.}
  \def\@permissionCodeOne{2378-962X}
}
%
%%
%% TeXSupport: Added new journal TSC on 2016/02/23
%%
\DeclareOption{acmtsc}{
  \typeout{}
  \typeout{Using ACM, TSC's option: 2016/02/23 by Aptara}
  \typeout{}
  \global\@acmjacmfalse
  \global\@acmtopsfalse
  \global\@acmtoclfalse
  \global\@acmtocsfalse
  \global\@acmtochifalse
  \global\@acmtodaesfalse
  \global\@acmtodsfalse
  \global\@acmtoisfalse
  \global\@acmtomacsfalse
  \global\@acmtomsfalse
  \global\@acmtoplasfalse
  \global\@acmtosemfalse
  \global\@acmtoitfalse
  \global\@acmtecsfalse
  \global\@acmtallipfalse
  \global\@acmjericfalse
  \global\@acmtacofalse
  \global\@acmjeafalse
  \global\@acmtslpfalse
  \global\@acmciefalse
  \global\@acmtosfalse
  \global\@acmcsurfalse
  \global\@acmjetcfalse
  \global\@acmtosnfalse
  \global\@acmtalgfalse
  \global\@acmtaasfalse
  \global\@acmtwebfalse
  \global\@acmtkddfalse
  \global\@acmtretsfalse
  \global\@acmtmisfalse
  \global\@acmtiisfalse
  \global\@acmtistfalse
  \global\@acmtoctfalse
  \global\@acmjdiqfalse
  \global\@acmtaccessfalse
  \global\@acmtallipfalse
  \global\@acmtiisfalse
  \global\@acmtocefalse
  \global\@acmteacfalse
  \global\@acmtommfalse
  \global\@acmtsasfalse
  \global\@acmtompecsfalse
  \global\@acmtcpsfalse
  \global\@acmtsctrue
  \def\@journalName{ACM Transactions on Social Computing}
  \def\@journalNameShort{ACM Trans. Soc. Comput.}
  \def\@permissionCodeOne{2469-7818}
}
%
\ExecuteOptions{twoside,final,10pt,letterpaper}
\ProcessOptions

\RequirePackage{ifpdf}%
\ifpdf
\pdfpagewidth=8.5in
\pdfpageheight=11in
\fi


% Including fonts for Production Mode
\ifprod@mode
% New Century Schoolbook as base font
\renewcommand\rmdefault{pnc}
% Helvetica as second font
\renewcommand\sfdefault{phv}
\fi

% Packages required
\RequirePackage{latexsym}
\RequirePackage{color}
\usepackage{graphicx}


\typeout{Document Class `acmsmall' Electronic Submissions
\fileversion\space <\filedate> (ACM).}

\def\acmVolume#1{\def\@acmVolume{#1}}
\def\acmNumber#1{\def\@acmNumber{#1}}
\def\acmArticle#1{\def\@acmArticle{#1}}
\def\articleSeq#1{\def\@articleSeq{#1}}
\def\acmPrice#1{\def\@acmPrice{#1}}
\def\acmYear#1{\def\@acmYear{#1}}
\def\acmMonth#1{\def\@acmMonth{#1}}
\def\doi#1{\def\@doi{#1}}


%To transform the month number in its name in English
\newcommand{\monthWord}[1]{\ifcase#1\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\else Month\fi}

\newcommand{\monthInf}[1]{\ifcase#1\or
  01\or 02\or 03\or 04\or 05\or 06\or
  07\or 08\or 09\or 10\or 11\or 12\else 00\fi}

% First and Last page
\newcount\@firstpg
\newcount\@lastpg
\def\lastpage#1{\global\advance\@lastpg#1\relax}
\AtEndDocument{\ifelec@app\else\immediate\write\@mainaux{\string\lastpage{\the\c@page}}\fi}

\newdimen\trimheight
\newdimen\trimwidth
\newdimen\normaltextheight
\newdimen\tempdimen  
\newbox\tempbox
\newbox\tbbox
\newdimen\tabledim

% Extra symbol
\DeclareSymbolFont{newlargesymbols}{OMX}{cmex}{m}{n}
\DeclareMathSymbol{\bigsqcup}{\mathop}{newlargesymbols}{"46}

% Bold Math
\def\boldmath{\mathversion{bold}}
\def\bm#1{\mathchoice
	  {\mbox{\boldmath$\displaystyle#1$}}%
          {\mbox{\boldmath$#1$}}%
          {\mbox{\boldmath$\scriptstyle#1$}}%
          {\mbox{\boldmath$\scriptscriptstyle#1$}}}

% Font information
\lineskip1\p@
\normallineskip1\p@
\def\baselinestretch{1}
\def\@ptsize{0} % needed for amssymbols.sty

\@maxdepth\maxdepth
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\newcommand{\pcal}{\@fontswitch{\relax}{\mathcal}}
\newcommand{\mit}{\protect\pmit}
\newcommand{\pmit}{\@fontswitch{\relax}{\mathnormal}}
\def\cal{\mathcal}

\renewcommand{\@ptsize}{}
\renewcommand{\normalsize}{%
  \@setfontsize\normalsize\@xpt{11\p@}%
  \abovedisplayskip .5\baselineskip \@plus2\p@ \@minus\p@
  \belowdisplayskip \abovedisplayskip
  \abovedisplayshortskip 6\p@ \@minus 3\p@
  \belowdisplayshortskip 6\p@ \@minus 3\p@
  \let\@listi\@listI
}
\newcommand{\small}{%
  \@setfontsize\small\@ixpt{10\p@}%
  \abovedisplayskip 5\p@ \@plus 2\p@ \@minus \p@
  \belowdisplayskip \abovedisplayskip
  \abovedisplayshortskip 5\p@ \@minus 2\p@
  \belowdisplayshortskip 5\p@ \@minus 2\p@
  \def\@listi{%
      \leftmargin\leftmargini
      \topsep 5\p@ \@plus 2\p@ \@minus .2\p@
      \parsep \z@ \@plus .7\p@
      \itemsep 1.6\p@ \@plus .8\p@}%
}%

\newcommand{\footnotesize}{%
    \@setfontsize\footnotesize\@viiipt{9\p@}%
    \abovedisplayskip 4\p@ \@plus \p@
    \belowdisplayskip \abovedisplayskip
    \abovedisplayshortskip 4\p@ \@minus \p@
    \belowdisplayshortskip 4\p@ \@minus \p@
    \def\@listi{%
        \leftmargin\leftmargini
        \topsep 4\p@ \@plus \p@
        \parsep \z@ \@plus .5\p@
        \itemsep \p@ \@plus .7\p@}
}%
\normalsize

\newcommand\scriptsize{\@setfontsize\scriptsize\@viipt\@viiipt}
\newcommand\tiny{\@setfontsize\tiny\@vpt\@vipt}
\newcommand\large{\@setfontsize\large\@xiipt{14}}
\newcommand\Large{\@setfontsize\Large\@xivpt{18}}
\newcommand\LARGE{\@setfontsize\LARGE\@xviipt{22}}
\newcommand\huge{\@setfontsize\huge\@xxpt{25}}
\newcommand\Huge{\@setfontsize\Huge\@xxvpt{30}}

\def\rhfont{\fontfamily{\sfdefault}\fontsize{9}{10}\selectfont}
\def\rfootfont{\fontsize{7}{8}\selectfont}
\def\foliofont{\fontfamily{\sfdefault}\fontsize{9}{10}\selectfont}

\def\sectionfont{\fontfamily{\sfdefault}\fontsize{9}{11}\selectfont\bfseries\raggedright}
\def\refsectionfont{\fontfamily{\sfdefault}\fontsize{9}{11}\selectfont\bfseries}
\def\subsectionfont{\fontfamily{\sfdefault}\fontsize{9}{11}\selectfont\bfseries\raggedright}
\def\subsubsectionfont{\fontfamily{\sfdefault}\fontsize{9}{11}\selectfont\itshape\raggedright}
\def\paragraphfont{\fontsize{10}{11}\selectfont\itshape}

\def\figcaptionfont{\fontsize{8}{9}\selectfont}%
\def\figcaptionnumfont{\fontfamily{\sfdefault}\fontsize{8}{9}\selectfont}%
\def\subcaptionfont{\fontsize{8}{10}\selectfont}%
\def\subcaption#1{{\centering\subcaptionfont#1\par}}

\def\tablefont{\fontsize{8}{9}\selectfont}%
\def\intexttablefont{\fontsize{8}{9}\selectfont\centering}%
\def\tablecaptionfont{\fontfamily{\sfdefault}\fontsize{8}{9}\selectfont}%
\def\tablenumfont{\fontfamily{\sfdefault}\fontsize{8}{9}\selectfont}%
\def\tabnotefont{\fontsize{8}{9}\selectfont}
\def\ackfont{\fontsize{8}{10}\selectfont}

\setlength\trimheight {10in}
\setlength\trimwidth  {6.75in}

\textheight 48pc
\advance\textheight-7.3pt
\setlength\normaltextheight{\textheight}

\textwidth 33pc
\oddsidemargin .625in
\evensidemargin\trimwidth
\advance\evensidemargin-\oddsidemargin
\advance\evensidemargin-\textwidth

\marginparwidth .5in
\marginparsep .125in
\topmargin 36pt
\headheight 6.5\p@
\topskip6.2pt
\headsep 17.5pt
\parindent10\p@
\newdimen\normalparindent
\normalparindent\parindent

\newlength{\footheight}%
\footheight 10\p@
\footskip 28\p@

\columnsep 12\p@
\columnseprule 0\p@

\footnotesep 7\p@
\skip\footins 15\p@ plus 4\p@ minus 3\p@
\floatsep 1\baselineskip plus 2\p@ minus 2\p@
\textfloatsep \floatsep
\intextsep 1pc plus 1pc

\newlength{\@maxsep}%
\@maxsep 1pc

\dblfloatsep 1\baselineskip plus 2\p@ minus 2\p@
\dbltextfloatsep 20\p@ plus 2\p@ minus 4\p@

\newlength{\@dblmaxsep}%
\@dblmaxsep 20\p@

\@fptop 0\p@ plus 1fil
\@fpsep 1pc plus 2fil
\@fpbot 0\p@ plus 1fil
\@dblfptop 0\p@ plus 1fil
\@dblfpsep 8\p@ plus 2fil
\@dblfpbot 0\p@ plus 1fil

\marginparpush 6\p@

\parskip0\p@
\partopsep 0\p@
\@lowpenalty 51
\@medpenalty 151
\@highpenalty 301

\@beginparpenalty -\@lowpenalty
\@endparpenalty -\@lowpenalty
\@itempenalty -\@lowpenalty

\setcounter{topnumber}{3}
\def\topfraction{.99}
\setcounter{bottomnumber}{1}
\def\bottomfraction{.5}
\setcounter{totalnumber}{3}
\def\textfraction{.01}
\def\floatpagefraction{.85}
\setcounter{dbltopnumber}{2}
\def\dbltopfraction{.7}
\def\dblfloatpagefraction{.5}

% ISSN
%
\def\issn#1{\gdef\@issn{#1}}
\issn{}

% Copyright Information
%
\RequirePackage{acmcopyright}

% Calculating total pages
\newcount\@totalpg
\def\acmPages#1{\def\@acmPages{#1}}
\acmPages{\@totalpg\@lastpg\global\advance\@totalpg-\@firstpg\global\advance\@totalpg\@ne\ifnum\the\@totalpg<\z@0\else\the\@totalpg\fi\ pages}

% Reference Format
\gdef\formatline{{\em{\@journalNameShort\ }}\@acmVolume, \@acmNumber, Article~\@acmArticle\ (\monthWord{\@acmMonth}\ \@acmYear), \@acmPages.\\ % Need Journal Name in italics - in 'ACM Reference Format' before 1. INTRODUCTION - Gerry - March 2012
{\tt DOI:} \@doi} % Gerry March 2012 - on first page in 'ACM Reference Format'
\gdef\copyrightline{%
  \if@printcopyright
  \copyright\ \@acmYear\ \@copyrightowner\  
  \else
  \@acmYear\ 
  \fi
  \@permissionCodeOne/\@acmYear/\monthInf{\@acmMonth}-ART\@acmArticle\ 
  \$\@acmPrice}
\gdef\doiline{{\tt DOI:} \@doi} % Gerry - March 2012 - beneath copyright line



% Page Style
\mark{{}{}}

\def\ps@headings{%
    \let\@mkboth\@gobbletwo
    \def\@oddhead{{\rhfont\rightmark}\hfill{\foliofont \@acmArticle:\thepage}}%
    \def\@evenhead{{\foliofont \@acmArticle:\thepage}\hfill{\rhfont\leftmark}}%
    \def\@evenfoot{\null\hfill{\rfootfont\@runningfoot}}
    \def\@oddfoot{{\rfootfont\@runningfoot} \hfill\null}%
    \let\partmark\@gobble
    \let\sectionmark\@gobble
    \let\subsectionmark\@gobble
}

\def\ps@appheadings{%
    \let\@mkboth\@gobbletwo
    \def\@oddhead{{\rhfont\rightmark}\hfill{\foliofont \thepage}}%
    \def\@evenhead{{\foliofont \thepage}\hfill{\rhfont\leftmark}}%
    \def\@evenfoot{\null\hfill{\rfootfont\@runningfoot}}
    \def\@oddfoot{{\rfootfont\@runningfoot} \hfill\null}%
    \let\partmark\@gobble
    \let\sectionmark\@gobble
    \let\subsectionmark\@gobble
}

\def\@runningfoot{}
\def\runningfoot#1{\gdef\@runningfoot{#1}}

\if@acmtodaes
 \def\pubphrase{Pub. date}
\else
\if@acmtosem
 \def\pubphrase{Pub. date}
\else
\if@acmjetc
 \def\pubphrase{Pub. date}
\else
\if@acmtiis
 \def\pubphrase{Pub. date}
\else
 \def\pubphrase{Publication date}
\fi\fi\fi\fi

\runningfoot{\@journalName, Vol.~\@acmVolume, No.~\@acmNumber, Article~\@acmArticle,\ \pubphrase:\ \monthWord{\@acmMonth}\ \@acmYear.}

\def\@firstfoot{}
\gdef\firstfoot{\@journalName, Vol.~\@acmVolume,\ No.~\@acmNumber,\ Article\ \@acmArticle,\ \pubphrase:\ \monthWord{\@acmMonth}\ \@acmYear.}

\def\ps@titlepage{%
    \let\@mkboth\@gobbletwo
    \let\@oddhead\@empty
    \def\@oddfoot{\null\hfill\fontsize{7}{8}\selectfont\firstfoot}
    \let\@evenhead\@empty
    \def\@evenfoot{\fontsize{7}{8}\selectfont\firstfoot\hfill\null}%
}

\def\titlepage{%
    \@restonecolfalse
    \if@twocolumn
        \@restonecoltrue
        \onecolumn
    \else
        \newpage
    \fi
    \thispagestyle{empty}%
    \c@page\z@
}

\def\endtitlepage{%
    \if@restonecol
        \twocolumn
    \else
        \newpage
    \fi
}

% Author and Affiliation
\def\author#1{\gdef\@author{#1}}%
\def\and{{\upshape and }}
\def\affil#1{\gdef\@affil{#1}\ifx\@affil\@empty\else{\reset@font\affilfont\unskip,\ #1\vphantom{gy}\endgraf}\fi}\affil{}

\def\titlefont{\fontfamily{\sfdefault}\fontsize{12}{12}\selectfont\bfseries}
\def\authorfont{\fontfamily{\sfdefault}\fontsize{10}{11}\rightskip0pt plus1fill\selectfont}
\def\affilfont{\fontfamily{\rmdefault}\fontsize{8}{11}\rightskip0pt plus1fill\selectfont}
\def\abstractfont{\fontsize{8}{10}\selectfont}

% Article Title page
\def\maketitle{%
    \newpage
    \thispagestyle{titlepage}%
    \global\@topnum\z@
    \begingroup
        \lineskip \z@
        \null
        \vskip -15.4\p@\relax
        \parindent\z@
        \begingroup
          \raggedright
          \hyphenpenalty\@M
          {\titlefont\@title\par}
	\global\@firstpg\the\c@page
        \endgroup
        \vskip 12\p@
        \begingroup

        {\addtolength{\baselineskip}{2\p@}%
            {\authorfont\@author\par}
            \vskip7pt
	    \ifx \@sponsors\@empty
	    \else
              \hbox{\vrule height .2\p@ width \textwidth}
              \@sponsors \par
            \fi
        }
\endgroup
     \par\vskip 22\p@\box\@abstract
     \par%
     \ifx\@categories\@empty 
     \else
        \abstractfont
         \vskip 4\p@\relax 
         \def\and{\unskip{\rm;} }%
         \noindent Categories and Subject Descriptors: \@categories\par
     \fi
     \ifx\@concepts\@empty % Added 2012/10/09
     \else
        \abstractfont
         \vskip 4\p@\relax 
         \noindent CCS Concepts: \@concepts\par
     \fi
     \ifx\@terms\@empty\else
        \abstractfont
        \vskip 4\p@
        \noindent  General Terms: \ignorespaces
        \@terms
        \par
     \fi
     \ifx\@keywords\@empty\else
        \abstractfont
	\vskip 4\p@
        \noindent  Additional Key Words and Phrases: \ignorespaces
        \@keywords
        \par
     \fi
     \ifx\@acmformat\@empty\else
        \abstractfont
        \vskip 4\p@
        \noindent  {\bf ACM Reference Format:}\\
        \@acmformat\vskip 0.5\p@
        \par
     \fi
    \par
    \endgroup
    \let\maketitle\relax
    \global\let\@sponsors\@empty
    \global\let\@categories\@empty
    \global\let\@concepts\@empty
    \global\let\@terms\@empty
    \global\let\@keywords\@empty
    }

% Sponsor
\def\sponsor#1{\@ifnextchar
 [{\@sponsor{#1}}{\@xsponsor{#1}}}
\def\@sponsor#1[#2]{\edef\@tempa{\ifx \@sponsors\@empty
 \else ; \fi}{\def\protect{\noexpand\protect
 \noexpand}\def\and{\noexpand\and}\xdef\@sponsors{\@sponsors\@tempa #1 }}}
\def\@xsponsor#1{\edef\@tempa{\ifx \@sponsors\@empty \else ;
\fi}{\def\protect{\noexpand\protect\noexpand}\def\and{\noexpand
 \and}\xdef\@sponsors{\@sponsors\@tempa #1}}}
\def\@sponsors{}

% Abstract
\newbox\@abstract
\def\abstract{
\global\setbox\@abstract=\vbox\bgroup\everypar{}%
\abstractfont\noindent\ignorespaces}
\def\endabstract{\egroup}

% Terms
\def\terms#1{\gdef\@terms{#1}}
\let\@terms\@empty

% Keywords
\def\keywords#1{\gdef\@keywords{#1}}
\let\@keywords\@empty

% Reference Format
\def\acmformat#1{\gdef\@acmformat{#1 \formatline\vphantom{y}}}
\let\@acmformat\@empty

% Category
\def\category#1#2#3{%
    \@ifnextchar [{\@category{#1}{#2}{#3}}{\@category{#1}{#2}{#3}[]}}
\def\@category#1#2#3[#4]{%
    \edef\@tempa{\ifx \@categories\@empty \else ; \fi}%
    \begingroup
        \def\protect{\noexpand\protect\noexpand}%
        \let\and\relax
        \xdef\@categories{%
            \@categories
            \@tempa
            #1 [{\bf #2}]%
            \if!#4!%
                \if!#3!\else : #3\fi
            \else
                :\space
                \if!#3!\else #3\kern\z@---\hskip\z@\fi
                {\it #4}%
            \fi
        }%
    \endgroup
}

% Support for CCSXML file
\RequirePackage{comment}
\excludecomment{CCSXML}

% New concepts scheme
\let\@concepts\@empty
%
% The first argument is the significance, the
% second is the concept(s)
%
\newcommand\ccsdesc[2][100]{%
  \ccsdesc@parse#1~#2~}
%
% The parser of the expression Significance~General~Specific
%
\def\ccsdesc@parse#1~#2~#3~{%
  \expandafter\ifx\csname CCS@#2\endcsname\relax
    \expandafter\gdef\csname CCS@#2\endcsname{\textbullet\textbf{#2} $\to$ }%
  \g@addto@macro{\@concepts}{\csname CCS@#2\endcsname}\fi
  \expandafter\g@addto@macro\expandafter{\csname CCS@#2\endcsname}{%
    \ifnum#1>499\textbf{#3; }\else
    \ifnum#1>299\textit{#3; }\else
    #3; \fi\fi}}

% Permission Information
\newbox\@permissionbox
\newenvironment{permission}{%
   \footnotesize
   \global\setbox\@permissionbox\vbox\bgroup\par\addvspace{3.1pt}\noindent\ignorespaces
   }{%
   \par\egroup}
%
\let\@categories\@empty
\def\bottomstuff{%
    \global\@topnum\z@
    \global\@botroom \textheight
    \@float{figure}[b]
        \footnotesize
        \parindent\z@
        \null
        \vskip -\textfloatsep
        \vskip 6\p@ plus2pt minus2pt
        \hrule \@height .5\p@ \@width \textwidth
        \vskip 6\p@ %7.5
        \ignorespaces
}
\def\endbottomstuff{\par%
  \@copyrightpermission
  \par\box\@permissionbox
  \copyrightline\par
  \doiline
  \vskip-13pt
  \strut
\end@float
}

% Heads
\newcounter{part}
\newcounter{section}
\newcounter{subsection}[section]
\newcounter{subsubsection}[subsection]
\newcounter{paragraph}[subsubsection]
\def\thepart{\Roman{part}}
\def\thesection{\arabic{section}}
\def\thesubsection{\thesection.\arabic{subsection}}
\def\thesubsubsection{\thesubsection.\arabic{subsubsection}}
\def\theparagraph{\thesubsubsection.\arabic{paragraph}}

% for checking Uppercase heads
\newif\if@uchead\@ucheadfalse

\setcounter{secnumdepth}{3}
\def\part{\@startsection{part}{9}{\z@}%
                        {-10\p@ \@plus -4\p@ \@minus -2\p@}%
                        {4\p@}%
                        {\normalsize\itshape\@ucheadtrue}}
\def\section{\@startsection{section}{1}{\z@}%
                           {-.75\baselineskip \@plus -2\p@ \@minus -.2\p@}%
                           {.25\baselineskip}%
			   {\sectionfont\@ucheadtrue}}
\def\refsection{\@startsection{section}{1}{\z@}%
                              {-1\baselineskip \@plus -2\p@ \@minus -.2\p@}%
                              {.5\baselineskip}%
			      {\refsectionfont\@ucheadtrue}}
\def\subsection{\@startsection{subsection}{2}{\z@}%
                              {-.75\baselineskip \@plus -2\p@ \@minus -.2\p@}%
			      {.25\baselineskip}%
			      {\subsectionfont}}
\def\subsubsection{\@startsection{subsubsection}{3}{10pt}%
                                 {-.5\baselineskip \@plus -2\p@ \@minus -.2\p@}%
				 {-3.5\p@}{\subsubsectionfont}}
\def\paragraph{\@startsection{paragraph}{4}{\parindent}%
                             {-.5\baselineskip \@plus -2\p@ \@minus -.2\p@}%
			     {-3.5\p@}%
			     {\paragraphfont}}

\def\@seccntformat#1{\csname the#1\endcsname.\hskip .4em}
\let\@period=.
\def\@sect#1#2#3#4#5#6[#7]#8{%
  \ifnum #2>\c@secnumdepth
    \let\@svsec\@empty
  \else
    \refstepcounter{#1}%
    \edef\@svsec{\begingroup\csname the#1\endcsname\endgroup\relax .\hskip .4em}%
    \protected@edef\@svsec{\@seccntformat{#1}\relax}%
  \fi
  \@tempskipa #5\relax
  \ifdim \@tempskipa>\z@
    \begingroup
      #6{%
        \@hangfrom{\hskip #3\relax\@svsec}%
          \interlinepenalty \@M \if@uchead\MakeUppercase{#8}\else#8\fi\@@par}%
    \endgroup
    \csname #1mark\endcsname{#7}%
    \addcontentsline{toc}{#1}{%
      \ifnum #2>\c@secnumdepth \else
        \protect\numberline{\csname the#1\endcsname}%
      \fi
      #7}%
  \else
    \def\@svsechd{%
      #6{\hskip #3\relax
      \@svsec #8}%
      \csname #1mark\endcsname{#7}%
      \addcontentsline{toc}{#1}{%
        \ifnum #2>\c@secnumdepth \else
          \protect\numberline{\csname the#1\endcsname}%
        \fi
        #7}}%
  \fi
  \@xsect{#5}}
\def\@xsect#1{%
    \@tempskipa #1\relax
    \ifdim \@tempskipa>\z@
        \par
        \nobreak
        \vskip \@tempskipa
        \@afterheading
    \else
        \global\@nobreakfalse
        \global\@noskipsectrue
        \everypar{%
            \if@noskipsec
                \global\@noskipsecfalse
                \clubpenalty\@M
                \hskip -\parindent
                \begingroup
                    \@svsechd
                    \@period
                \endgroup
                \unskip
                \@tempskipa #1\relax
                \hskip -\@tempskipa
            \else
                \clubpenalty \@clubpenalty
                \everypar{}%
            \fi
        }%
    \fi
    \ignorespaces
}

% Lists
\def\@trivlist{%
    \@topsepadd\topsep
    \if@noskipsec
        \global\let\@period\@empty
        \leavevmode
        \global\let\@period.%
    \fi
    \ifvmode
        \advance\@topsepadd\partopsep
    \else
        \unskip
        \par
    \fi
    \if@inlabel
        \@noparitemtrue
        \@noparlisttrue
    \else
        \@noparlistfalse
        \@topsep\@topsepadd
    \fi
    \advance\@topsep \parskip
    \leftskip\z@skip
    \rightskip\@rightskip
    \parfillskip\@flushglue
    \@setpar{\if@newlist\else{\@@par}\fi}
    \global\@newlisttrue
    \@outerparskip\parskip
}
\labelsep 5\p@

\settowidth{\leftmargini}{9.}
\addtolength\leftmargini\labelsep

\settowidth{\leftmarginii}{(b)}
\addtolength\leftmarginii\labelsep

\leftmarginiii  \leftmarginii
\leftmarginiv   \leftmarginii
\leftmarginv    \leftmarginii
\leftmarginvi   \leftmarginii
\leftmargin     \leftmargini

\labelwidth\leftmargini
\advance\labelwidth-\labelsep
\def\@listI{%
    \leftmargin\leftmargini
    \parsep \z@ 
    \topsep .5\baselineskip \@plus 2\p@%
    \itemsep\z@%
}

\let\@listi\@listI

\@listi
\def\@listii{%
    \leftmargin\leftmarginii
    \labelwidth\leftmarginii
    \advance\labelwidth-\labelsep
    \topsep \z@ 
    \parsep \z@ 
    \itemsep \parsep
}
\def\@listiii{%
    \leftmargin\leftmarginiii
    \labelwidth\leftmarginiii
    \advance\labelwidth-\labelsep
    \topsep \z@ \@plus \p@
    \parsep \z@ 
    \itemsep \parsep
}
\def\@listiv{%
    \leftmargin\leftmarginiv
    \labelwidth\leftmarginiv
    \advance\labelwidth-\labelsep
}
\def\@listv{%
    \leftmargin\leftmarginv
    \labelwidth\leftmarginv
    \advance\labelwidth-\labelsep
}
\def\@listvi{%
    \leftmargin\leftmarginvi
    \labelwidth\leftmarginvi
    \advance\labelwidth-\labelsep
}
\newdimen\enumdim
\def\enummax#1{%
  \setbox\tempbox\hbox{#1\hskip\labelsep}%
  \enumdim\wd\tempbox
  \expandafter\global\csname leftmargin\romannumeral\the\@enumdepth\endcsname
  \enumdim}

\enummax{1.}
\def\enumerate{\@ifnextchar[{\@enumerate}{\@enumerate[\csname label\@enumctr\endcsname]}}%% 
\def\@enumerate[#1]{\par\abovedisplayskip .25\baselineskip \@plus2\p@
                    \belowdisplayskip .25\baselineskip \@plus2\p@
      \ifnum \@enumdepth >3 \@toodeep\else
      \advance\@enumdepth\@ne
      \edef\@enumctr{enum\romannumeral\the\@enumdepth}%
      \setcounter{\@enumctr}{1}\enummax{#1}%
      \list
      {\csname label\@enumctr\endcsname}{\usecounter{\@enumctr}%
      \def\makelabel##1{\hss\llap{##1}}}\fi}
\def\endenumerate{\endlist}%% 
\def\longenum{%
    \leftmargin0pt
    \ifnum \@enumdepth >3
        \@toodeep
    \else
        \advance\@enumdepth \@ne
        \edef\@enumctr{enum\romannumeral\the\@enumdepth}%
        \list{\csname label\@enumctr\endcsname}{%
            \usecounter{\@enumctr}%
            \labelwidth\z@\leftmargin0pt
	    \itemindent\parindent\advance\itemindent\labelsep
        }%
    \fi
}
\let\endlongenum\endlist

\def\labelenumi{{\rm (}\arabic{enumi}\/{\rm )}} 
\def\theenumi{\arabic{enumi}} 
\def\labelenumii{{\rm (}\alph{enumii}\rm{)}}
\def\theenumii{\alph{enumii}}
\def\p@enumii{\theenumi}
\def\labelenumiii{\roman{enumiii}.}
\def\theenumiii{\roman{enumiii}}
\def\p@enumiii{\theenumi{\rm (}\theenumii{\rm )}}
\def\labelenumiv{\Alph{enumiv}.}
\def\theenumiv{\Alph{enumiv}}
\def\p@enumiv{\p@enumiii\theenumiii}
\newcommand\labelitemi{\texthyphen}%
\newcommand\labelitemii{\texthyphen}%
\newcommand\labelitemiii{\texthyphen}%
\newcommand\labelitemiv{\texthyphen}%
\font\lcir = lcircle10 at 8pt
\newcommand\bulls{\raise1.5pt\hbox{\lcir\char'162}}
\def\textbullet{\leavevmode\raise2pt\hbox{\bulls}}
\def\texthyphen{---}
\newif\ifhyphen\hyphenfalse

\def\itemize{%
  \ifnum \@itemdepth >3 \@toodeep\else \advance\@itemdepth \@ne
  \edef\@itemitem{labelitem\romannumeral\the\@itemdepth}%
  \list{\csname\@itemitem\endcsname}%
       {\advance\leftmargin-5.7pt\labelsep2pt
       \def\makelabel##1{\hss\llap{##1}}}\fi}%
\let\enditemize =\endlist

\newenvironment{hyphenlist}{%
   \hyphentrue
   \renewcommand\labelitemi{\texthyphen}%
   \renewcommand\labelitemii{\texthyphen}%
   \renewcommand\labelitemiii{\texthyphen}%
   \renewcommand\labelitemiv{\texthyphen}%
   \begin{itemize}
}{%
   \end{itemize}
   \hyphenfalse}
%
\newenvironment{bulletlist}{%
\leftmargini20pt
   \renewcommand\labelitemi{\textbullet}%
   \renewcommand\labelitemii{\textbullet}%
   \renewcommand\labelitemiii{\textbullet}%
   \renewcommand\labelitemiv{\textbullet}%
   \begin{itemize}
}{%
   \end{itemize}}

\def\longitem{%
    \list{---}{%
        \labelwidth\z@
        \leftmargin\z@
        \itemindent\parindent
        \advance\itemindent\labelsep
    }%
}
\let\endlongitem\endlist
\def\verse{%
    \let\\=\@centercr
    \list{}{%
        \leftmargin 2pc
        \itemindent -1.5em
        \listparindent \itemindent
        \rightmargin\leftmargin
        \advance\leftmargin 1.5em
    }%
    \item[]
}

\let\endverse\endlist
\def\quotation{%
    \list{}{%
        \leftmargin 2pc
        \listparindent .5em
        \itemindent\listparindent
        \rightmargin\leftmargin
        \parsep \z@ \@plus \p@
    }%
    \item[]
}

\let\endquotation=\endlist
\def\quote{%
    \list{}{%
        \leftmargin 2pc
        \rightmargin\leftmargin
    }%
    \item[]
}

\let\endquote=\endlist
%
\def\extractfont{\fontsize{8}{10}\selectfont\leftskip8pt\rightskip8pt}
\newenvironment{extract}{\par\addvspace{7pt plus2pt}
\parindent8pt
\extractfont
\noindent\ignorespaces
}{%
\par\addvspace{6pt plus2pt}}

\def\descriptionlabel#1{%
  \hspace\labelsep \normalfont\itshape #1.%
}%
\newenvironment{description}{%
 \list{}{%
    \labelwidth\z@
    \let\makelabel\descriptionlabel
 }
}{\endlist}%

\def\describe#1{%
    \list{}{%
        \listparindent\parindent
        \settowidth{\labelwidth}{#1}%
        \leftmargin\labelwidth
        \addtolength\leftmargin\labelsep
        \def\makelabel##1{##1\hfil}%
    }%
}
\let\enddescribe\endlist

\def\program{%
    \ifx\@currsize\normalsize
        \small
    \else
        \rm
    \fi
    \tabbing
}
\let\endprogram\endtabbing

% Enunciations
\newif\ifitalicenv\italicenvtrue

\newtheorem{theorem}{Theorem}[section]
\newtheorem{conjecture}[theorem]{Conjecture}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
%
\newtheorem{exam}[theorem]{Example}
\newenvironment{example}{%
\italicenvfalse
\begin{exam}}{\end{exam}\italicenvtrue}
%
\newtheorem{defi}[theorem]{Definition}
\newenvironment{definition}{%
\italicenvfalse
\begin{defi}}{\end{defi}\italicenvtrue}

\def\@begintheorem#1#2{%
    \trivlist
    \item[%
        \hskip 12\p@
        \hskip \labelsep
        {\ifitalicenv\sc\else\itshape\fi #1\hskip 5\p@\relax{\rm #2}.\enspace}]%
        \ifitalicenv\itshape\else\upshape\fi\hskip-\labelsep%
}
\def\@opargbegintheorem#1#2#3{%
    \trivlist
    \item[\hskip 12pt
          \hskip \labelsep
          {\ifitalicenv{\sc{#1}}\else{\itshape#1}\fi%
	   \savebox\@tempboxa{\ifitalicenv{\scshape#3}\else{\itshape#3}\fi}%
	   \ifdim\wd\@tempboxa>\z@%
           \ {\rm #2}\unskip\hskip5pt\relax$($\box\@tempboxa$)$% 
	   \fi.\unskip\hskip5pt}]
\ifitalicenv\itshape\else\upshape\fi\hskip-\labelsep}

\newif\if@qeded
\global\@qededfalse
\def\proof{%
    \global\@qededfalse
    \@ifnextchar[{\@xproof}{\@proof}}

\def\endproof{%
    \if@qeded\else\qed\fi
    \endtrivlist
}
\def\@proof{%
    \trivlist
    \item[%
        \hskip 12\p@
        \hskip \labelsep
        {\sc Proof.\enspace}]\hskip-\labelsep%
    \ignorespaces
}
\def\@xproof[#1]{%
    \trivlist
    \item[\hskip 12\p@\hskip \labelsep{\sc Proof #1.}]%
    \ignorespaces
}
\def\qed{\unskip\kern 10pt{\unitlength1pt\linethickness{.4pt}\framebox(5,5){}}
    \global\@qededtrue
    }%
\def\newdef#1#2{%
    \expandafter\@ifdefinable\csname #1\endcsname
        {\@definecounter{#1}%
         \expandafter\xdef\csname the#1\endcsname{\@thmcounter{#1}}%
         \global\@namedef{#1}{\@defthm{#1}{#2}}%
         \global\@namedef{end#1}{\@endtheorem}%
    }%
}
\def\@defthm#1#2{%
    \refstepcounter{#1}%
    \@ifnextchar[{\@ydefthm{#1}{#2}}{\@xdefthm{#1}{#2}}%
}
\def\@xdefthm#1#2{%
    \@begindef{#2}{\csname the#1\endcsname}%
    \ignorespaces
}
\def\@ydefthm#1#2[#3]{%
    \trivlist
    \item[%
        \hskip 10\p@
        \hskip \labelsep
        {\it #2%
         \savebox\@tempboxa{#3}%
         \ifdim \wd\@tempboxa>\z@
            \ \box\@tempboxa
         \fi.%
        }]%
    \ignorespaces
}
\def\@begindef#1#2{%
    \trivlist
    \item[%
        \hskip 10\p@
        \hskip \labelsep
        {\it #1\ \rm #2.}%
    ]%
}

% Maths
\def\theequation{\arabic{equation}}

\def\@marrayclassiv{\@addtopreamble{$\displaystyle \@nextchar$}}
\def\@marrayclassz{\ifcase \@lastchclass \@acolampacol \or \@ampacol \or
 \or \or \@addamp \or
 \@acolampacol \or \@firstampfalse \@acol \fi
\edef\@preamble{\@preamble
 \ifcase \@chnum 
 \hfil$\relax\displaystyle\@sharp$\hfil \or $\relax\displaystyle\@sharp$\hfil 
 \or \hfil$\relax\displaystyle\@sharp$\fi}}
\def\marray{\arraycolsep 2.5pt\let\@acol\@arrayacol \let\@classz\@marrayclassz
 \let\@classiv\@marrayclassiv \let\\\@arraycr\def\@halignto{}\@tabarray}
\def\endmarray{\crcr\egroup\egroup}

\arraycolsep 2.5\p@
\tabcolsep 6\p@
\arrayrulewidth .25\p@
\doublerulesep 2\p@
\tabbingsep \labelsep

\skip\@mpfootins = \skip\footins
\fboxsep = 3pt \fboxrule = .4\p@

% Table of Contents
\def\@pnumwidth{1.55em}
\def\@tocrmarg {2.55em}
\def\@dotsep{4.5}
\setcounter{tocdepth}{3}
\def\tableofcontents{%
    \section*{Contents\@mkboth{CONTENTS}{CONTENTS}}
    \@starttoc{toc}%
}
\def\l@part#1#2{%
    \addpenalty\@secpenalty
    \addvspace{2.25em \@plus \p@}%
    \begingroup
        \@tempdima 3em
        \parindent \z@
        \rightskip \@pnumwidth
        \parfillskip -\@pnumwidth
        {\large \bf \leavevmode #1\hfil \hb@xt@\@pnumwidth{\hss #2}}\par
        \nobreak
        \endgroup
}
\def\l@section#1#2{%
    \addpenalty\@secpenalty
    \addvspace{1.0em \@plus \p@}%
    \@tempdima 1.5em
    \begingroup
        \parindent \z@
        \rightskip \@pnumwidth
        \parfillskip -\@pnumwidth
        \bf \leavevmode #1\hfil \hb@xt@\@pnumwidth{\hss #2}\par
    \endgroup
}
\def\l@subsection{\@dottedtocline{2}{1.5em}{2.3em}}
\def\l@subsubsection{\@dottedtocline{3}{3.8em}{3.2em}}

% List of figures
\def\listoffigures{%
    \section*{List of Figures\@mkboth{LIST OF FIGURES}{LIST OF FIGURES}}
    \@starttoc{lof}}
\def\l@figure{\@dottedtocline{1}{1.5em}{2.3em}}

% List of tables
\def\listoftables{%
    \section*{List of Tables\@mkboth{LIST OF TABLES}{LIST OF TABLES}}
    \@starttoc{lot}}
\let\l@table\l@figure

% Index
\newif\if@restonecol
\def\theindex{%
    \@restonecoltrue
    \if@twocolumn \@restonecolfalse \fi
    \columnseprule \z@
    \columnsep 35\p@
    \twocolumn[\section*{Index}]
    \@mkboth{INDEX}{INDEX}%
    \thispagestyle{plain}%
    \parindent\z@
    \parskip\z@ \@plus .3\p@\relax
    \let\item\@idxitem
}

\def\endtheindex{%
    \if@restonecol
        \onecolumn
    \else
        \clearpage
    \fi
}
\def\@idxitem{\par\hangindent 40\p@}
\def\subitem{\par\hangindent 40\p@ \hspace*{20\p@}}
\def\subsubitem{\par\hangindent 40\p@ \hspace*{30\p@}}
\def\indexspace{\par \vskip 10\p@ \@plus 5\p@ \@minus 3\p@\relax}

% Footnote
\def\footnoterule{%
    \kern-3\p@
    \hrule \@width 36\p@ \@height 0.25\p@
    \kern 3\p@
}
\long\def\@makefntext#1{%
    \parindent 1em
    \noindent $^{\@thefnmark}$#1%
}

% Figures and Tables
\newlength\belowcaptionskip
\newlength\abovecaptionskip
\setlength\belowcaptionskip{3.5\p@}
\setlength\abovecaptionskip{6\p@}

\def\nocaption{\refstepcounter\@captype \par
 \vskip 1pc \hbox to\hsize{\hfil \figcaptionnumfont Figure \thefigure
 \hfil}}

\def\FigName{figure}%
\long\def\@caption#1[#2]#3{\par\begingroup
    \@parboxrestore
    \normalsize
    \@makecaption{\csname fnum@#1\endcsname}{\ignorespaces #3}\par
  \endgroup}

\long\def\@makecaption#1#2{%
  \ifx\FigName\@captype\baselineskip10pt
    \vskip\abovecaptionskip
    \setbox\@tempboxa\hbox{\figcaptionfont{\figcaptionnumfont#1}.\hskip7.3pt\relax #2\par}%
    \ifdim \wd\@tempboxa >\hsize
        \figcaptionfont{\figcaptionnumfont#1}.\hskip7.3pt\relax #2\par
    \else
        \centerline{\box\@tempboxa}%
    \fi
  \else%
    \setbox\tbbox=\vbox{\hsize\tempdimen\tablecaptionfont{\tablenumfont #1.\ }{#2\par}}%
    \setbox\@tempboxa\hbox{\hsize\tempdimen\tablecaptionfont{\tablenumfont #1.}\ {#2\par}}%
    \ifdim \wd\@tempboxa >\tempdimen
        \centerline{\box\tbbox}%        
    \else
        \centerline{\box\@tempboxa}%
    \fi
    \vskip\belowcaptionskip
  \fi
}
%
\newcounter{figure}
\def\thefigure{\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
%
\newif\ifcontinued
\global\continuedfalse
%
\def\continued{\global\continuedtrue}
%
\def\fnum@figure{\ifcontinued\global\continuedfalse\addtocounter{figure}{-1} Fig.~\thefigure---{\it Continued}\else Fig.~\thefigure\fi}%
\newenvironment{figure}
	       {\@float{figure}}
	       {\end@float}
\newenvironment{figure*}
	       {\@dblfloat{figure}}
	       {\end@dblfloat}
%
\newcounter{table}
\renewcommand\thetable{\Roman{table}}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\tablename{Table}
\def\fnum@table{\ifcontinued\addtocounter{table}{-1} Table~\thetable---{\it Continued} \else Table~\thetable\fi}
\newenvironment{table}
               {\@float{table}}
               {\end@float}
\newenvironment{table*}
               {\@dblfloat{table}}
               {\end@dblfloat}

\long\def\tbl#1#2{%
\setbox\tempbox\hbox{\tablefont #2}%
\tabledim\hsize\advance\tabledim by -\wd\tempbox
\tempdimen\wd\tempbox
\global\divide\tabledim\tw@
\caption{#1\protect\vphantom{yp}}
\centerline{\box\tempbox}}%

\newenvironment{intexttable}{\par\addvspace{1.2pt plus2pt}
\intexttablefont
}{%
\par\addvspace{7pt plus2pt}}

\newenvironment{tabnote}{%
\par\vskip5pt
\tabnotefont
\@ifnextchar[{\@tabnote}{\@tabnote[]}}{%
\par}
\def\@tabnote[#1]{\def\@Tempa{#1}\leftskip\tabledim\rightskip\leftskip\ifx\@Tempa\@empty\else{\it #1:}\ \fi\ignorespaces}

\def\tabnoteentry#1#2{\parindent0pt\par{#1}{#2}}
\def\Note#1#2{\parindent0pt\par{\it #1}\ #2}

\newdimen\@narrowfig
\newbox\@nfigbox
\newbox\@nfcapbox
\newif\if@nfeven
\def\acmtable#1{%
    \@narrowfig #1\relax
    \let\caption\@atcap
    \let\nocaption\@atnocap
    \@ifnextchar[{\@ntab}{\@ntab[\fps@table]}%
}

\def\@ntab[#1]{%
    \expandafter\table\expandafter[#1]%
    \setbox\@nfigbox\vbox\bgroup
        \hsize \@narrowfig
        \@parboxrestore
}

\def\endacmtable{%
    \hb@xt@\textwidth{%
        \hfil
        \vbox{%
            \hsize \@narrowfig
            \box\@nfcapbox{%
                \baselineskip 4\p@
                \hbox{\vrule \@height .4\p@ \@width \hsize}%

            }%
            \vskip -\p@
            \box\@nfigbox
            \vskip -\p@
            \begingroup
                \baselineskip 4\p@
                \hbox{\vrule \@height .4\p@ \@width \hsize}%
            \endgroup
        }%
        \hfil
    }%
    \end@float
}
\def\@atmakecap#1#2{%
    \setbox\@tempboxa\hbox{#1.\hskip 1em\relax #2}%
    \ifdim \wd\@tempboxa >\hsize
        \sloppy #1.\hskip 1em\relax #2 \par
    \else
        \centerline{\box\@tempboxa}%
    \fi
}
\def\@atcap{%
        \par
    \egroup
    \refstepcounter\@captype
    \@dblarg{\@atcapx\@captype}%
}
\long\def\@atcapx#1[#2]#3{%
    \addcontentsline{\csname ext@#1\endcsname}{#1}{%
        \protect\numberline{\csname the#1\endcsname}{\ignorespaces #2}%
    }%
    \setbox\@nfcapbox\vbox{%
        \hsize \wd\@nfigbox
        \@parboxrestore
        \@atmakecap{\csname fnum@#1\endcsname}{\ignorespaces #3}%
        \par
    }%
}
\def\@atnocap{%
    \egroup
    \refstepcounter\@captype
    \setbox\@nfcapbox\vbox{%
        \hsize \wd\@nfigbox
        \centerline{\footnotesize \fnum@table}
    }%
}
\def\narrowfig#1{%
    \@narrowfig #1\relax
    \let\caption\@nfcap
    \let\nocaption\@nfnocap
    \@ifnextchar[{\@nfig}{\@nfig[\fps@figure]}%
}

\def\@nfig[#1]{%
    \expandafter\figure\expandafter[#1]%
    \setbox\@nfigbox\vbox\bgroup
        \hsize\@narrowfig
        \@parboxrestore
}

\def\endnarrowfig{%
    \hb@xt@\textwidth{%
        \if@nfeven
            \box\@nfcapbox  \hfil   \box\@nfigbox
        \else
            \box\@nfigbox   \hfil   \box\@nfcapbox
        \fi
    }%
    \endfigure
}
\def\@nfcap{%
        \par
    \egroup
    \refstepcounter\@captype
    \@dblarg{\@nfcapx\@captype}%
}
\def\@nfmakecap #1#2{%
    \setbox\@tempboxa\hbox{#1.\hskip 1em\relax #2}%
    \ifdim \wd\@tempboxa >\hsize
        \sloppy #1.\hskip 1em\relax #2 \par
    \else
        \@@line{%
            \if@nfeven\else\hfil\fi
            \box\@tempboxa
            \if@nfeven\hfil\fi
        }%
    \fi
}
\long\def\@nfcapx#1[#2]#3{%
    \addcontentsline{\csname ext@#1\endcsname}{#1}{%
        \protect\numberline{\csname the#1\endcsname}{\ignorespaces #2}%
    }%
    \@seteven
    \setbox\@nfcapbox\vbox to \ht\@nfigbox{%
        \hsize \textwidth
        \advance\hsize -2pc
        \advance\hsize -\wd\@nfigbox
        \@parboxrestore
        \vfil
        \@nfmakecap{\csname fnum@#1\endcsname}{\ignorespaces #3}%
        \par
        \vfil
    }%
}
\def\@nfnocap{%
    \egroup
    \refstepcounter\@captype
    \@seteven
    \setbox\@nfcapbox\vbox to \ht\@nfigbox{%
        \hsize \textwidth
        \advance\hsize -2pc
        \advance\hsize -\wd\@nfigbox
        \@parboxrestore
        \vfil
        \@@line{%
            \if@nfeven\else\hfil\fi
            \footnotesize \fnum@figure
            \if@nfeven\hfil\fi
        }%
        \vfil
    }%
}

\def\@seteven{%
    \@nfeventrue
    \@ifundefined{r@@nf\thefigure}{}{%
        \edef\@tmpnf{\csname r@@nf\thefigure\endcsname}%
        \edef\@tmpnf{\expandafter\@cdr\@tmpnf\@nil}%
        \ifodd\@tmpnf\relax
            \@nfevenfalse
        \fi
    }%
    \label{@nf\thefigure}%
    \edef\@tmpnfx{\if@nfeven e\else o\fi}%
    \edef\@tmpnf{%
        \write\@unused{%
            \noexpand\ifodd \noexpand\c@page
                \noexpand\if \@tmpnfx e%
                    \noexpand\@nfmsg{\thefigure}
                \noexpand\fi
            \noexpand\else
                \noexpand\if \@tmpnfx o%
                    \noexpand\@nfmsg{\thefigure}%
                \noexpand\fi
            \noexpand\fi
        }%
    }%
    \@tmpnf
}

\def\@nfmsg#1{Bad narrowfig: Figure #1 on page \thepage}
\newbox\@nfigbox
\newbox\@nfcapbox
\newif\if@nfeven

% Acknowledgments
\newenvironment{ack}{%
\refsection*{ACKNOWLEDGMENT}
\ackfont
}{\par}

\newenvironment{acks}{%
\refsection*{ACKNOWLEDGMENTS}
\ackfont
}{\par}

% History Dates
\def\received#1#2#3{\par\addvspace{14\p@}%
{\noindent\fontfamily{\sfdefault}\fontsize{8}{9}\selectfont{Received\ #1;\ revised\ #2;\ accepted\ #3}%
\par}}

% provide both spellings of Acknowledgment(s)
\let\acknowledgments\acks
\let\endacknowledgments\endacks
\let\acknowledgment\ack
\let\endacknowledgment\endack

\newcommand{\longpage}{\enlargethispage{\baselineskip}}
\newcommand{\shortpage}{\enlargethispage{-\baselineskip}}

% Appendices
\newfont{\apbf}{cmbx9}
\def\@withappendix#1{App--\number #1}
\newcommand{\elecappendix}{
}

\def\appenheader{\global\@topnum\z@ \global\@botroom \textheight \begin{figure}
\newfont{\sc}{cmcsc10}
\parindent\z@
\hbox{}
\vskip -\textfloatsep
\vskip 11pt
\hrule height .2pt width 30pc
\vskip 2pt\rule{0pt}{10pt}\ignorespaces}
\def\endappenheader{\end{figure}\gdef\appendixhead{}}

\def\@appsec{}

\def\appendix{\par
 \setcounter{section}{0}
 \setcounter{subsection}{0}
 \def\@appsec{APPENDIX } 
        \def\thesection{\Alph{section}}
        \def\theHsection{\Alph{section}}}

% split electronic appendix into two parts:

\def\appendixhead#1{\appendix
\section*{ELECTRONIC APPENDIX}
The electronic appendix for this article can be accessed in the ACM Digital Library.}

\newif\ifelec@app\global\elec@appfalse
\long\def\elecappendix{\immediate\write\@mainaux{\string\lastpage{\the\c@page}}%
\clearpage
\makeatletter
\elec@apptrue
\pagenumbering{withappendix}
\pagestyle{appheadings}
\thispagestyle{titlepage}%
\makeatother
\appendix
{\vbox{\titlefont\parindent0pt\raggedright%
Online Appendix to:\par\@title\par{}}}\vskip 12\p@
\vbox{\authorfont\parindent0pt\@author{}}\vskip .5em\noindent
\vskip 11pt\noindent
\hrule height .2pt
\par
\def\endbottomstuff{%
\copyrightline\par
\doiline
\vskip-13pt
\strut
\end@float
}
\bottomstuff
\endbottomstuff
}

%-----------------------BIBLIOGRAPHY STUFF-------------------------
% this is adapted (November 1993) by Andrew Appel and Rebecca Davies from
% 
%%%     filename        = "chicago.sty",
%%%     version         = "4",  % MODIFIED!
%%%     date            = "31 August 1992",
%%%     time            = "09:42:44 199",
%%%     author          = "Glenn Paulley",
%%%     address         = "Data Structuring Group
%%%                        Department of Computer Science
%%%                        University of Waterloo
%%%                        Waterloo, Ontario, Canada
%%%                        N2L 3G1",
%%%     telephone       = "(519) 885-1211",
%%%     FAX             = "(519) 885-1208",
%%%     email           = "gnpaulle@bluebox.uwaterloo.ca",

%%% ====================================================================
%
% this file: Modification of chicago.sty for new ACM bibliography
% style, which is similar (but not identical) to the ``Chicago'' style.
%
% chicago.sty: Style file for use with bibtex style chicago.bst, for
% bibliographies formatted according to the 13th Edition of the Chicago
% Manual of Style.
%
% 'newapa.bst' was made from 'plain.bst', 'named.bst', and 'apalike.bst',
% with lots of tweaking to make it look like APA style, along with tips
% from Young Ryu and Brian Reiser's modifications of 'apalike.bst'.
% newapa.sty formed the basis of this style, chicago.sty. Author-date
% references in newapa.bst formed the basis for chicago.bst. Chicagoa.bst
% supports annotations.
%
% Version 4 (August, 1992):
% - fixed chicago.bst and chicagoa.bst to handle long author lists in
%   sorting
% - fixed chicago.bst and chicagoa.bst so that missing page numbers in
%   ``article'' entries are handled correctly
% - modified chicago.sty to format entries with 2nd and subsequent lines
%   indented.
%
%   Citation format: (author-last-name year)
%             (author-last-name and author-last-name year)
%             (author-last-name et al. year)
%             (author-last-name)
%             author-last-name
%             author-last-name (year)
%             (author-last-name and author-last-name)
%             (author-last-name et al.)
%             (year) or (year,year)
%             year or year,year
%
%   Reference list ordering: alphabetical by author or whatever passes
%    for author in the absence of one.
%
% This BibTeX style has support for abbreviated author lists and for
%    year-only citations.  This is done by having the citations
%    actually look like
%
%    \citeauthoryear{full-author-info}{abbrev-author-info}{year}
%
% The LaTeX style has to have the following (or similar)
%
%     \let\@internalcite\cite
%     \def\fullcite{\def\citeauthoryear##1##2##3{##1, ##3}\@internalcite}
%     \def\fullciteA{\def\citeauthoryear##1##2##3{##1}\@internalcite}
%     \def\shortcite{\def\citeauthoryear##1##2##3{##2, ##3}\@internalcite}
%     \def\shortciteA{\def\citeauthoryear##1##2##3{##2}\@internalcite}
%     \def\citeyear{\def\citeauthoryear##1##2##3{##3}\@internalcite}
%
%
% -------------------------------------------------------------------------
%
% Citation macros (compatible with natbib package).
%
\AtBeginDocument{%
\@ifpackageloaded{natbib}{% Natbib loaded
\renewcommand\bibsection{%
    \refsection*{{\refname}%
        \@mkboth{\uppercase{\refname}}{\uppercase{\refname}}%
}}%
\bibpunct[, ]{[}{]}{;}{a}{}{,}
\let\citeN\cite
\let\cite\citep
\let\citeANP\citeauthor
\let\citeNN\citeyearpar
\let\citeyearNP\citeyear
\let\citeyear\citeyearpar
\let\citeNP\citealt
\def\shortcite#1{\citeyear{#1}}
\DeclareRobustCommand\citeA
   {\begingroup\NAT@swafalse\let\NAT@ctype\@ne\NAT@partrue\NAT@fullfalse\NAT@open\NAT@citetp}
\newcommand\newblock{}
}{% Natbib not loaded
\let\@internalcite\cite
\def\cite{\def\@citeseppen{-1000}%
    \def\@cite##1##2{[##1\if@tempswa , ##2\fi]}%
    \def\citeauthoryear##1##2##3{##2 ##3}\@internalcite}
\def\citeXNP#1#2{%
  \def\@cite@label{#2}%
  \def\@cite##1##2{##1\if@tempswa , ##2\fi}%
  \def\citeauthoryear##1##2##3{\@cite@label}\@internalcite{#1}}
\def\citeNP{\def\@citeseppen{-1000}%
    \def\@cite##1##2{##1\if@tempswa , ##2\fi}%
    \def\citeauthoryear##1##2##3{##2 ##3}\@internalcite}
\def\citeN{\def\@citeseppen{-1000}%
    \def\@cite##1##2{##1\if@tempswa , ##2]\else{]}\fi}%
    \def\citeauthoryear##1##2##3{##2 [##3}\@citedata}
\def\shortcite#1{\citeyear{#1}}
\def\citeS#1{[\citeANP{#1} \citeyearNP{#1}]}
\def\citeNS#1{\citeANP{#1} \citeyear{#1}}
\def\citeNPS#1{\citeANP{#1} \citeyearNP{#1}}
%testing year,year
\def\citeNN{\def\@citeseppen{-1000}%
    \def\@cite##1##2{[##1\if@tempswa , ##2\fi]}%
    \def\citeauthoryear##1##2##3{##3}\@citedata}

\def\citeA{\def\@citeseppen{-1000}%
    \def\@cite##1##2{[##1\if@tempswa , ##2\fi]}%
    \def\citeauthoryear##1##2##3{##2}\@internalcite}
\def\citeANP{\def\@citeseppen{-1000}%
    \def\@cite##1##2{##1\if@tempswa , ##2\fi}%
    \def\citeauthoryear##1##2##3{##2}\@internalcite}

\def\citeyear{\def\@citeseppen{-1000}%
    \def\@cite##1##2{[##1\if@tempswa , ##2\fi]}%
    \def\citeauthoryear##1##2##3{##3}\@citedata}
\def\citeyearNP{\def\@citeseppen{-1000}%
    \def\@cite##1##2{##1\if@tempswa , ##2\fi}%
    \def\citeauthoryear##1##2##3{##3}\@citedata}

%
% \@citedata and \@citedatax:
%
% Place commas in-between citations in the same \citeyear, \citeyearNP,
% or \citeN command.
% Use something like \citeN{ref1,ref2,ref3} and \citeN{ref4} for a list.
%
\def\@citedata{%
        \@ifnextchar [{\@tempswatrue\@citedatax}%
                                  {\@tempswafalse\@citedatax[]}%
}

\def\@citedatax[#1]#2{%
\if@filesw\immediate\write\@auxout{\string\citation{#2}}\fi%
  \def\@citea{}\@cite{\@for\@citeb:=#2\do%
    {\@citea\def\@citea{; }\@ifundefined% by Young
       {b@\@citeb}{{\bf ?}%
       \@warning{Citation `\@citeb' on page \thepage \space undefined}}%
{\csname b@\@citeb\endcsname}}}{#1}}%

% don't box citations, separate with ; and a space
% also, make the penalty between citations negative: a good place to break.
%
\def\@citex[#1]#2{%
\if@filesw\immediate\write\@auxout{\string\citation{#2}}\fi%
  \def\@citea{}\@cite{\@for\@citeb:=#2\do%
    {\@citea\def\@citea{; }\@ifundefined% by Young
       {b@\@citeb}{{\bf ?}%
       \@warning{Citation `\@citeb' on page \thepage \space undefined}}%
{\csname b@\@citeb\endcsname}}}{#1}}%

% Fix cite so it doesn't repeat author lists in citations:

\def\cite{\def\@citeseppen{-1000}%
    \def\@cite##1##2{[##1\if@tempswa , ##2\fi]}%
    \let\@lastauthor=\@noauthor
    \let\citeauthoryear=\citeauthoryear@no@repeats\@internalcite}
    %\def\citeauthoryear##1##2##3{##2 ##3}\@internalcite

\def\@noauthor={\relax}
\let\@lastauthor=\@noauthor
\let\@currauthor=\@noauthor

\def\citeauthoryear@no@repeats#1#2#3{%
  \def\@currauthor{\csname @author #1\endcsname}%
  \ifx\@lastauthor\@currauthor{#3}\else{#2 #3}\fi%
  \let\@lastauthor=\@currauthor}
}}

% Bibliography
\let\@biblabel\@gobble
\newdimen\bibindent
\bibindent=16pt
\newcommand\refname{REFERENCES}
\def\thebibliography#1{%
    \footnotesize
    \refsection*{{\refname}
        \@mkboth{\uppercase{\refname}}{\uppercase{\refname}}%
    }
    \list{}{
        \settowidth\labelwidth{}
        \leftmargin0pt
        \advance\leftmargin\bibindent
        \itemindent -\bibindent
        \itemsep2pt
        \parsep \z@
        \usecounter{enumi}%
    }%
    \let\newblock\@empty
    \sloppy
    \sfcode`\.=1000\relax
}
\let\endthebibliography=\endlist

% Blackboard font 
\DeclareFontFamily{OT1}{ams}{}
\DeclareFontShape{OT1}{ams}{m}{n}{ <-> msam10 }{}
\DeclareFontShape{OT1}{ams}{m}{it}{ <-> msam10 }{}
\DeclareFontShape{OT1}{ams}{bx}{n}{ <-> msbm10 }{}
\DeclareFontShape{OT1}{ams}{bx}{it}{ <-> msbm10 }{}

\def\bb#1{\mathchoice
	 {\mbox{\fontfamily{ams}\fontsize{\tf@size}{\tf@size}\selectfont\bf#1}}%
	 {\mbox{\fontfamily{ams}\fontsize{\tf@size}{\tf@size}\selectfont\bf#1}}%
	 {\mbox{\fontfamily{ams}\fontsize{\sf@size}{\sf@size}\selectfont\bf#1}}%
	 {\mbox{\fontfamily{ams}\fontsize{\ssf@size}{\ssf@size}\selectfont\bf#1}}}

\DeclareMathAlphabet{\mathbb}{OT1}{ams}{bx}{n}
\SetMathAlphabet\mathbb{normal}{OT1}{ams}{bx}{n}
\SetMathAlphabet\mathbb{bold}{OT1}{ams}{bx}{n}
\def\bb{\mathbb}

\def\text#1{\ifmmode
        \mathchoice
	{\hbox{\fontsize{\tf@size}{\tf@size}\selectfont#1}}%
	{\hbox{\fontsize{\tf@size}{\tf@size}\selectfont#1}}%
	{\hbox{\fontsize{\sf@size}{\sf@size}\selectfont#1}}%
	{\hbox{\fontsize{\ssf@size}{\ssf@size}\selectfont#1}}%
       \else\hbox{\rm#1}\fi}
%
% newdef need to take the optional parameters of newtheorem
\def\newdef#1{\@ifnextchar[{\@xnewdef{#1}}{\@ynewdef{#1}}}
\def\@xnewdef#1[#2]#3{\newtheorem{italic@#1}[#2]{{\em #3}}\@newdef{#1}}
\def\@ynewdef#1#2{\@ifnextchar[{\@xynewdef{#1}{#2}}{\@yynewdef{#1}{#2}}}
\def\@xynewdef#1#2[#3]{\newtheorem{italic@#1}{{\em #2}}[#3]\@newdef{#1}}
\def\@yynewdef#1#2{\newtheorem{italic@#1}{{\em #2}}\@newdef{#1}}
% 
\def\@newdef#1{\newenvironment{#1}{\@ifnextchar[{\@xstartdef{#1}}{\@ystartdef{#1}}}{\end{italic@#1}}}
%
\def\@xstartdef#1[#2]{\begin{italic@#1}[{\em #2}]\rm}
\def\@ystartdef#1{\begin{italic@#1}\rm}
%
\newdef{remark}[theorem]{Remark}

% Page number Panel
\def\openartnum{\vbox to 1.125in{\vskip.125in\vfill\hbox to 26pt{\textcolor{white}{\fontfamily{\sfdefault}\fontsize{12}{7}\selectfont\bfseries\@acmArticle}\hfill}\vfill}}
\def\closeartnum{\vbox to 1.125in{\vskip-.125in\vfill\hbox to 26pt{\textcolor{white}{\fontfamily{\sfdefault}\fontsize{12}{7}\selectfont\bfseries\@acmArticle}\hfill}\vfill}}
\def\artnum{\vbox to 1in{\vfill\hbox to 26pt{\textcolor{white}{\fontfamily{\sfdefault}\fontsize{12}{7}\selectfont\bfseries\@acmArticle}\hfill}\vfill}}
%
\def\opentab{\hbox{\vrule width45.75pt depth0pt height1.125in\hspace*{-35.5pt}\openartnum}}
\def\normtab{\hbox{\vrule width45.75pt depth0pt height1in\hspace*{-35.5pt}\artnum}}
\def\closetab{\hbox{\vrule width45.75pt depth0pt height1.125in\hspace*{-35.5pt}\closeartnum}}
%
\newcommand\tab{\ifcase\@articleSeq\or
\vbox{\vskip-.125in\opentab}\or \vbox{\vskip1in\normtab}\or \vbox{\vskip2in\normtab}\or \vbox{\vskip3in\normtab}\or \vbox{\vskip4in\normtab}\or \vbox{\vskip5in\normtab}\or \vbox{\vskip6in\normtab}\or \vbox{\vskip7in\normtab}\or \vbox{\vskip8in\normtab}\or \vbox{\vskip9in\closetab}\or
\vbox{\vskip-.125in\opentab}\or \vbox{\vskip1in\normtab}\or \vbox{\vskip2in\normtab}\or \vbox{\vskip3in\normtab}\or \vbox{\vskip4in\normtab}\or \vbox{\vskip5in\normtab}\or \vbox{\vskip6in\normtab}\or \vbox{\vskip7in\normtab}\or \vbox{\vskip8in\normtab}\or \vbox{\vskip9in\closetab}\or
\vbox{\vskip-.125in\opentab}\or \vbox{\vskip1in\normtab}\or \vbox{\vskip2in\normtab}\or \vbox{\vskip3in\normtab}\or \vbox{\vskip4in\normtab}\or \vbox{\vskip5in\normtab}\or \vbox{\vskip6in\normtab}\or \vbox{\vskip7in\normtab}\or \vbox{\vskip8in\normtab}\or \vbox{\vskip9in\closetab}\fi}

% Output Routine
\def\@outputpage{%
\begingroup           % the \endgroup is put in by \aftergroup
  \let \protect \noexpand
  \@resetactivechars
  \@parboxrestore
  \shipout \vbox{%
    \set@typeset@protect
    \aftergroup \endgroup
    \aftergroup \set@typeset@protect
                                % correct? or just restore by ending
                                % the group?
  \if@specialpage
    \global\@specialpagefalse\@nameuse{ps@\@specialstyle}%
  \fi
  \if@twoside
    \ifodd\count\z@ \let\@thehead\@oddhead \let\@thefoot\@oddfoot
         \let\@themargin\oddsidemargin
    \else \let\@thehead\@evenhead
       \let\@thefoot\@evenfoot \let\@themargin\evensidemargin
    \fi
  \fi
  \reset@font
  \normalsize
  \normalsfcodes
  \let\label\@gobble
  \let\index\@gobble
  \let\glossary\@gobble
  \baselineskip\z@skip \lineskip\z@skip \lineskiplimit\z@
    \@begindvi
    \ifelec@app\else\ifnum\c@page=1\vbox to 0pt{\hbox to \trimwidth{\hfill\tab}}\fi\fi
    \vskip \topmargin
    \moveright\@themargin \vbox {%
      \setbox\@tempboxa \vbox to\headheight{%
        \vfil
        \color@hbox
          \normalcolor
          \hb@xt@\textwidth{\@thehead}%
        \color@endbox
        }%                        %% 22 Feb 87
      \dp\@tempboxa \z@
      \box\@tempboxa
      \vskip \headsep
      \box\@outputbox
      \baselineskip \footskip
      \color@hbox
        \normalcolor
        \hb@xt@\textwidth{\@thefoot}%
      \color@endbox
      }%
    }%
  \global \@colht \textheight
  \stepcounter{page}%
  \let\firstmark\botmark
}
%
\ps@headings
\pagenumbering{arabic}
\onecolumn

\frenchspacing
\sloppy
\widowpenalty10000
\clubpenalty10000

\endinput

% End of file `acmsmall.cls' Version 1.4


