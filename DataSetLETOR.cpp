/*
 * LETOR.cpp
 *
 *  Created on: Feb 4, 2012
 *      Author: wchan
 */

#include "DataSetLETOR.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

RankSet DataSetLETOR::m_train;
RankSet DataSetLETOR::m_test;

// TODO: right now it only uses fold1: this needs to be updated later...

void DataSetLETOR::load() {
  std::ifstream train("letor/Fold1/train.txt", std::ifstream::in | std::ifstream::binary);
  std::ifstream test("letor/Fold1/test.txt", std::ifstream::in | std::ifstream::binary);
  std::ifstream validate("letor/Fold1/validate.txt", std::ifstream::in | std::ifstream::binary);

  m_train.clear();
  m_test.clear();

  for (std::size_t i = 0; i < 10000 && train.good(); i++) {
    std::size_t query;
    boost::numeric::ublas::vector<double> features;
    double label;

    features.resize(get_feature_size(), false);

    // 2 qid:1 1:3 2:3 3:0 4:0 5:3 6:1 7:1 8:0 9:0 10:1 11:156 12:4 13:0 14:7 15:167 16:6.931275 17:22.076928 18:19.673353 19:22.255383 20:6.926551 21:3 22:3 23:0 24:0 25:6 26:1 27:1 28:0 29:0 30:2 31:1 32:1 33:0 34:0 35:2 36:1 37:1 38:0 39:0 40:2 41:0 42:0 43:0 44:0 45:0 46:0.019231 47:0.75000 48:0 49:0 50:0.035928 51:0.00641 52:0.25000 53:0 54:0 55:0.011976 56:0.00641 57:0.25000 58:0 59:0 60:0.011976 61:0.00641 62:0.25000 63:0 64:0 65:0.011976 66:0 67:0 68:0 69:0 70:0 71:6.931275 72:22.076928 73:0 74:0 75:13.853103 76:1.152128 77:5.99246 78:0 79:0 80:2.297197 81:3.078917 82:8.517343 83:0 84:0 85:6.156595 86:2.310425 87:7.358976 88:0 89:0 90:4.617701 91:0.694726 92:1.084169 93:0 94:0 95:2.78795 96:1 97:1 98:0 99:0 100:1 101:1 102:1 103:0 104:0 105:1 106:12.941469 107:20.59276 108:0 109:0 110:16.766961 111:-18.567793 112:-7.760072 113:-20.838749 114:-25.436074 115:-14.518523 116:-21.710022 117:-21.339609 118:-24.497864 119:-27.690319 120:-20.203779 121:-15.449379 122:-4.474452 123:-23.634899 124:-28.119826 125:-13.581932 126:3 127:62 128:11089534 129:2 130:116 131:64034 132:13 133:3 134:0 135:0 136:0
    std::string line;
    getline(train, line);

    boost::algorithm::trim(line);
    std::vector<std::string> tokens;
    boost::split(tokens, line, boost::is_any_of(" "));

    assert(tokens.size() == 138);

    label = boost::lexical_cast<double>(tokens[0]);
    query = boost::lexical_cast<std::size_t>(tokens[1].substr(tokens[1].find(":") + 1, tokens[1].size()));

    for (std::size_t i = 2; i < tokens.size(); i++) {
      features[i - 2] = boost::lexical_cast<double>(tokens[i].substr(tokens[i].find(":") + 1, tokens[i].size()));
    }

    m_train.push_back(query, features, label);

    train.peek();
  }

  train.close();
  test.close();
  validate.close();
}

std::size_t DataSetLETOR::get_feature_size() {
  return 136;
}

std::size_t DataSetLETOR::get_label_size() {
  return 1;
}

RankSet DataSetLETOR::get_train() {
  if (m_train.size() == 0) {
    load();
  }

  return m_train;
}

RankSet DataSetLETOR::get_test() {
  if (m_test.size() == 0) {
    load();
  }
  return m_test;
}
