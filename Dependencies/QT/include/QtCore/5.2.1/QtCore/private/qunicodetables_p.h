/****************************************************************************
**
** Copyright (C) 2013 Digia Plc and/or its subsidiary(-ies).
** Contact: http://www.qt-project.org/legal
**
** This file is part of the QtCore module of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:LGPL$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and Digia.  For licensing terms and
** conditions see http://qt.digia.com/licensing.  For further information
** use the contact form at http://qt.digia.com/contact-us.
**
** GNU Lesser General Public License Usage
** Alternatively, this file may be used under the terms of the GNU Lesser
** General Public License version 2.1 as published by the Free Software
** Foundation and appearing in the file LICENSE.LGPL included in the
** packaging of this file.  Please review the following information to
** ensure the GNU Lesser General Public License version 2.1 requirements
** will be met: http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
**
** In addition, as a special exception, Digia gives you certain additional
** rights.  These rights are described in the Digia Qt LGPL Exception
** version 1.1, included in the file LGPL_EXCEPTION.txt in this package.
**
** GNU General Public License Usage
** Alternatively, this file may be used under the terms of the GNU
** General Public License version 3.0 as published by the Free Software
** Foundation and appearing in the file LICENSE.GPL included in the
** packaging of this file.  Please review the following information to
** ensure the GNU General Public License version 3.0 requirements will be
** met: http://www.gnu.org/copyleft/gpl.html.
**
**
** $QT_END_LICENSE$
**
****************************************************************************/

/* This file is autogenerated from the Unicode 6.2 database. Do not edit */

//
//  W A R N I N G
//  -------------
//
// This file is not part of the Qt API.  It exists for the convenience
// of internal files.  This header file may change from version to version
// without notice, or even be removed.
//
// We mean it.
//

#ifndef QUNICODETABLES_P_H
#define QUNICODETABLES_P_H

#include <QtCore/qchar.h>

QT_BEGIN_NAMESPACE

#define UNICODE_DATA_VERSION QChar::Unicode_6_2

namespace QUnicodeTables {

struct Properties {
    ushort category            : 8; /* 5 used */
    ushort direction           : 8; /* 5 used */
    ushort combiningClass      : 8;
    ushort joining             : 2;
    signed short digitValue    : 6; /* 5 used */
    signed short mirrorDiff    : 16;
    signed short lowerCaseDiff : 16;
    signed short upperCaseDiff : 16;
    signed short titleCaseDiff : 16;
    signed short caseFoldDiff  : 16;
    ushort lowerCaseSpecial    : 1;
    ushort upperCaseSpecial    : 1;
    ushort titleCaseSpecial    : 1;
    ushort caseFoldSpecial     : 1;
    ushort unicodeVersion      : 4;
    ushort nfQuickCheck        : 8;
    ushort graphemeBreakClass  : 4; /* 4 used */
    ushort wordBreakClass      : 4; /* 4 used */
    ushort sentenceBreakClass  : 8; /* 4 used */
    ushort lineBreakClass      : 8; /* 6 used */
    ushort script              : 8; /* 7 used */
};

Q_CORE_EXPORT const Properties * QT_FASTCALL properties(uint ucs4);
Q_CORE_EXPORT const Properties * QT_FASTCALL properties(ushort ucs2);

enum GraphemeBreakClass {
    GraphemeBreak_Other,
    GraphemeBreak_CR,
    GraphemeBreak_LF,
    GraphemeBreak_Control,
    GraphemeBreak_Extend,
    GraphemeBreak_RegionalIndicator,
    GraphemeBreak_Prepend,
    GraphemeBreak_SpacingMark,
    GraphemeBreak_L,
    GraphemeBreak_V,
    GraphemeBreak_T,
    GraphemeBreak_LV,
    GraphemeBreak_LVT
};

enum WordBreakClass {
    WordBreak_Other,
    WordBreak_CR,
    WordBreak_LF,
    WordBreak_Newline,
    WordBreak_Extend,
    WordBreak_RegionalIndicator,
    WordBreak_Katakana,
    WordBreak_ALetter,
    WordBreak_MidNumLet,
    WordBreak_MidLetter,
    WordBreak_MidNum,
    WordBreak_Numeric,
    WordBreak_ExtendNumLet
};

enum SentenceBreakClass {
    SentenceBreak_Other,
    SentenceBreak_CR,
    SentenceBreak_LF,
    SentenceBreak_Sep,
    SentenceBreak_Extend,
    SentenceBreak_Sp,
    SentenceBreak_Lower,
    SentenceBreak_Upper,
    SentenceBreak_OLetter,
    SentenceBreak_Numeric,
    SentenceBreak_ATerm,
    SentenceBreak_SContinue,
    SentenceBreak_STerm,
    SentenceBreak_Close
};

// see http://www.unicode.org/reports/tr14/tr14-30.html
// we don't use the XX and AI classes and map them to AL instead.
enum LineBreakClass {
    LineBreak_OP, LineBreak_CL, LineBreak_CP, LineBreak_QU, LineBreak_GL,
    LineBreak_NS, LineBreak_EX, LineBreak_SY, LineBreak_IS, LineBreak_PR,
    LineBreak_PO, LineBreak_NU, LineBreak_AL, LineBreak_HL, LineBreak_ID,
    LineBreak_IN, LineBreak_HY, LineBreak_BA, LineBreak_BB, LineBreak_B2,
    LineBreak_ZW, LineBreak_CM, LineBreak_WJ, LineBreak_H2, LineBreak_H3,
    LineBreak_JL, LineBreak_JV, LineBreak_JT, LineBreak_RI, LineBreak_CB,
    LineBreak_SA, LineBreak_SG, LineBreak_SP, LineBreak_CR, LineBreak_LF,
    LineBreak_BK
};

Q_CORE_EXPORT GraphemeBreakClass QT_FASTCALL graphemeBreakClass(uint ucs4);
inline GraphemeBreakClass graphemeBreakClass(QChar ch)
{ return graphemeBreakClass(ch.unicode()); }

Q_CORE_EXPORT WordBreakClass QT_FASTCALL wordBreakClass(uint ucs4);
inline WordBreakClass wordBreakClass(QChar ch)
{ return wordBreakClass(ch.unicode()); }

Q_CORE_EXPORT SentenceBreakClass QT_FASTCALL sentenceBreakClass(uint ucs4);
inline SentenceBreakClass sentenceBreakClass(QChar ch)
{ return sentenceBreakClass(ch.unicode()); }

Q_CORE_EXPORT LineBreakClass QT_FASTCALL lineBreakClass(uint ucs4);
inline LineBreakClass lineBreakClass(QChar ch)
{ return lineBreakClass(ch.unicode()); }

} // namespace QUnicodeTables

QT_END_NAMESPACE

#endif // QUNICODETABLES_P_H
