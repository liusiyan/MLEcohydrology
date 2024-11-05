program driver

   implicit none

   real, external:: nn1, nn2
   !print *, nn1(0.0000000e+00, 1.7703299e+01, 1.0000000e+00, 3.5869876e+05, 2.4686513e+01, 2.9124221e+01)
   print *, nn2(3.42600000e-02, 2.66055000e+01, 1.93500000e-01, 6.79296378e+05, 3.84000000e+00, 3.20210000e+01)
   ! 25.278332
end program driver

function nn1(lmr_z, par_z, rh_can, gb_mol, je, cair) result(x0)

   implicit none
   real, intent(in):: lmr_z, par_z, rh_can, gb_mol, je, cair
   real:: x0

   real:: mean(6) = (/0.466418, 79.152236, 0.595526, 720561.017006, 46.180845, 29.111094/)
   real:: std(6) = (/0.290595, 71.098227, 0.241566, 167301.360669, 33.213463, 0.171379/)
   real:: w1(6, 6) = reshape((/-0.01188245, -0.29741922, -0.44035792, -0.22048871, 0.9688138, &
      -0.15193653, 3.04251, 0.13223447, 1.6849937, 0.46811637, &
      -2.7942166, 0.32345018, -1.4623866, 0.42111543, -0.95796543, &
      -0.19300316, -2.445707, -0.06666169, -0.09045773, -0.1397287, &
      1.3354509, 0.11165843, 3.4360676, -0.04416294, 0.1925234, &
      -0.39150804, -2.5510962, 0.3881439, 0.4748766, 0.1708479, &
      -2.109834, 0.2984501, -1.8083875, -0.25520384, 1.0789176, &
      -0.2167491/), (/6, 6/))
   real:: b1(6) = (/-1.3167274, 1.3226284, -2.79317, 4.4891477, -4.6568084, 0.00925752/)
   real:: w2(6) = (/-2.463204, -2.6419995, 2.4009244, 3.353584, -3.515716, -2.8274136/)
   real:: b2 = 0.1744493
   real:: x(6), y
   integer:: i, j
   x = (/lmr_z, par_z, rh_can, gb_mol, je, cair/)
   ! normalize
   do i = 1, 6
      x(i) = (x(i) - mean(i))/std(i)
   end do
   x0 = 0.0
   ! a 6-layer with sigmoid activation
   do i = 1, 6
      y = 0.0
      do j = 1, 6
         y = y + x(j)*w1(j, i)
      end do
      y = 1./(1.+exp(-(y + b1(i))))
      ! a 1-layer with linear activation
      x0 = x0 + w2(i)*y
   end do
   x0 = x0 + b2
   ! un-standardize
   x0 = 2.379772*x0 + 21.861976
end function nn1

function nn2(lmr_z, par_z, rh_can, gb_mol, je, cair) result(x0)

   implicit none
   real, intent(in):: lmr_z, par_z, rh_can, gb_mol, je, cair
   real:: x0

   real:: mean(6) = (/0.5387583374977112, 122.86979675292969, 0.6250501275062561, 769434.6875, 55.70592498779297, &
      33.124210357666016/)
   real:: std(6) = (/0.21232753992080688,86.79254913330078,0.21232452988624573,227883.03125,31.62569808959961, &
      2.962263345718384/)
   real:: w1(6, 64) = reshape((/2.4506771564483643, -0.657319962978363, 2.9716081619262695, -0.3639841079711914, &
      -1.1000391244888306, 0.1493874341249466, 3.5409998893737793, -1.3067891597747803, 6.433949947357178, 2.2014105319976807, &
      -3.357102632522583, -0.19104668498039246, -7.060011863708496, 3.298579216003418, -4.407839775085449, -2.046971321105957, &
      10.046504020690918, 0.2819439172744751, -0.5047094225883484, -0.9393593668937683, 3.43743896484375, -0.12528705596923828, &
      1.7851989269256592, -0.1914805918931961, -2.8890233039855957, 0.4376387298107147, -1.6278423070907593, -0.08827657997608185, &
      -3.2394449710845947, -0.7435671091079712, -0.4171352684497833, -0.08954884856939316, -0.033410411328077316, &
      0.29106655716896057, 0.18460695445537567, 6.413644313812256, -0.1399279534816742, 0.21384389698505402, 0.28750038146972656, &
      0.36954203248023987, -3.497860908508301, 2.845369815826416, 0.3753809928894043, 0.23161348700523376, -2.6566548347473145, &
      -0.4441966116428375, -2.813786506652832, 3.848511219024658, 2.3758699893951416, 0.5196965932846069, 0.5291990041732788, &
      -0.513434648513794, -7.8701300621032715, -0.4251601994037628, -1.0796188116073608, 0.23509925603866577, 4.306929588317871, &
      -0.7433184385299683, -2.76103138923645, -0.6511070728302002, -6.987130641937256, 4.310287952423096, -2.8646438121795654, &
      -2.784133195877075, 12.21593952178955, 0.31219300627708435, 0.7776870131492615, 0.047200579196214676, 0.3767167329788208, &
      -0.7305867671966553, -0.36037346720695496, -5.02765417098999, 1.7328510284423828, 1.0404046773910522, -1.1083394289016724, &
      -2.1592905521392822, -7.028836250305176, -0.46530240774154663, 0.8089573383331299, -1.1455615758895874, -3.6151154041290283, &
      -5.008387088775635, 5.589188098907471, 0.1720675826072693, 0.5604009032249451, 0.849495530128479, 3.371443033218384, &
      3.063387870788574, -4.546784400939941, -0.27380064129829407, -7.429487228393555, -0.5724537372589111, -2.299701452255249, &
      -0.5655125975608826, 13.165989875793457, 0.39579108357429504, 0.12611114978790283, -0.207645833492279, -1.20302212238311, &
      -1.453136920928955, -0.9036486744880676, 1.6607379913330078, 4.3010945320129395, 0.22690360248088837, 1.414833903312683, &
      2.0985031127929688, -6.788771152496338, -0.016162650659680367, 3.0176289081573486, -0.48905494809150696, 4.465543746948242, &
      2.0138940811157227, -3.844371795654297, -0.10094565153121948, -4.043070316314697, -0.6458062529563904, 5.623471736907959, &
      -0.8257729411125183, 6.103351593017578, -0.30244964361190796, 6.516400337219238, 0.009848798625171185, 2.19195556640625, &
      2.7118513584136963, -7.200827121734619, 0.28465089201927185, -0.23104286193847656, -0.1081279069185257, 4.534897327423096, &
      5.145016193389893, -1.0454363822937012, -0.220394566655159, -0.2794252932071686, -0.05979600176215172, -0.06957044452428818, &
      0.23894208669662476, 0.059757083654403687, 4.3191986083984375, 0.052902866154909134, -0.029587673023343086, &
      1.135772705078125, 0.48698219656944275, 1.0107545852661133, 2.819340229034424, -2.0857677459716797, -0.2571132481098175, &
      1.5489616394042969, 0.8577113151550293, 0.287222295999527, 2.1672685146331787, 0.8230421543121338, 0.7728961706161499, &
      0.5326831936836243, -1.68449068069458, -3.7479727268218994, -0.3269859552383423, -1.068516492843628, 6.943055629730225, &
      1.3680022954940796, 2.821500778198242, -5.706265449523926, 0.026904482394456863, -0.46473202109336853, 0.2407827526330948, &
      -0.05742431804537773, 1.2799125909805298, -0.14203335344791412, 7.225398063659668, 2.1239047050476074, -1.0485203266143799, &
      1.504088044166565, -0.47354960441589355, -1.6861727237701416, -0.10315540432929993, -1.5911805629730225, 3.1087820529937744, &
      2.3133883476257324, 0.799994170665741, 4.693713188171387, 0.3901527523994446, -0.5113613605499268, -0.8540899157524109, &
      7.777887344360352, 0.5102351307868958, 1.8349772691726685, -0.35572853684425354, 3.9210050106048584, -3.0085935592651367, &
      0.6564633250236511, 2.1444344520568848, -8.032851219177246, -0.2320970594882965, -3.626372814178467, 4.070871829986572, &
      -0.15325714647769928, 0.45806559920310974, 8.311165809631348, 0.770118236541748, 0.9883197546005249, -0.4229795038700104, &
      1.22611665725708, -0.9537543058395386, 1.2164795398712158, 1.1463932991027832, -0.2897670269012451, 1.255708932876587, &
      2.658527135848999, -1.3825918436050415, -3.103818416595459, -0.4212844669818878, 0.44234389066696167, -0.742725670337677, &
      2.2155046463012695, 2.206641674041748, 6.487067222595215, 1.4491804838180542, -0.5329986810684204, 0.25316858291625977, &
      -0.027198132127523422, 1.3644623756408691, -0.05614104121923447, 8.38369083404541, -6.56557559967041, -0.12776945531368256, &
      -2.192351818084717, -1.343955159187317, 6.397023677825928, -0.1620836853981018, -0.1721440702676773, -0.6755569577217102, &
      2.1611456871032715, 1.120712161064148, -1.2760401964187622, -0.14949394762516022, -8.6624116897583, -0.35415714979171753, &
      -2.8725662231445312, -1.02963125705719, 11.373934745788574, 0.18864886462688446, -0.7206297516822815, 0.8674324750900269, &
      -4.393677711486816, 1.7327388525009155, -0.09724074602127075, 0.46971264481544495, -0.45931264758110046, -4.994523525238037, &
      -3.2954983711242676, 1.798042893409729, -4.287132263183594, -0.32301652431488037, 0.5275014638900757, 0.21188658475875854, &
      -5.898828983306885, -2.976016044616699, -0.48502588272094727, 0.25782471895217896, -3.1118247509002686, -0.1861472874879837, &
      -4.167741298675537, -2.7408275604248047, 4.856125831604004, -0.00752479350194335, -0.32954561710357666, 0.9354353547096252, &
      1.263893961906433, 2.6287240982055664, -4.091916084289551, -0.15099477767944336, -0.966986894607544, 0.6397818326950073, &
      -3.097066640853882, -1.9628349542617798, -1.3756613731384277, -0.36026817560195923, 0.6811145544052124, 0.1123274639248848, &
      0.6435600519180298, 0.12174402177333832, -7.941482067108154, 0.13788780570030212, 0.9496744275093079, -0.15826815366744995, &
      2.9967830181121826, -4.845188617706299, 2.512608289718628, -0.36662644147872925, -2.4221811294555664, 0.14980275928974152, &
      0.45858824253082275, 0.5905314683914185, -1.6498974561691284, 1.8733837604522705, -3.2479147911071777, 0.8909579515457153, &
      4.37186336517334, -0.9514419436454773, 2.645263671875, -0.19291283190250397, -4.5243730545043945, -0.3811015486717224, &
      0.2946180999279022, 0.39983123540878296, 1.1605974435806274, 1.026305913925171, -0.7100013494491577, 0.1280670315027237, &
      -0.23256933689117432, 1.2438422441482544, 0.4504172205924988, 4.593224048614502, 10.735027313232422, -0.2614177465438843, &
      2.9281444549560547, 0.66932213306427, -11.934906959533691, -0.12405146658420563, 7.099161148071289, 0.34187525510787964, &
      1.7312381267547607, 0.5620031952857971, -14.076203346252441, -0.5995762944221497, -2.4085376262664795, 0.8351727724075317, &
      1.2182376384735107, 0.6626801490783691, -0.3798445165157318, 0.6942718029022217, 1.2056409120559692, 0.2148808389902115, &
      0.11317740380764008, 0.0917125940322876, -10.201458930969238, -0.07938782125711441, 0.64609283208847, -0.42517516016960, &
      2.140990734100342, -0.9858496785163879, 1.9920079708099365, 0.6588010191917419, -0.5871385335922241, 0.238653302192688, &
      -0.11240105330944061, 1.3297698497772217, 0.22612854838371277, 5.472451686859131, -1.0192744731903076, -1.1737912893295288, &
      0.6531755328178406, 1.0909614562988281, -2.7305147647857666, 0.5410414934158325, -0.14331233501434326, -0.6167135834693909, &
      1.0911195278167725, -0.7046239972114563, 8.337404251098633, 1.2597492933273315, 3.299426317214966, 0.5316376090049744, &
      1.813879370689392, 1.0665831565856934, 0.17519782483577728, 0.4857149124145508, -2.2876710891723633, -0.19256985187530518, &
      0.9632585644721985, 0.6279392242431641, -0.16733112931251526, 2.116943120956421, -6.02890157699585, 1.1883624792099, &
      -4.023552894592285, -1.7618519067764282, 4.779090404510498, -0.31335899233818054, 3.5427169799804688, -0.6199512481689453, &
      7.522680759429932, -0.7146159410476685, -1.9694671630859375, 0.06816954910755157/), (/6, 64/))
   real:: b1(64) = (/2.0093657970428467, 7.441710948944092, -7.69439172744751, 3.455004930496216, -9.620753288269043, &
      5.262630462646484, -11.710822105407715, -9.582902908325195, -8.523711204528809, 11.871475219726562, -3.849119186401367, &
      -7.679129123687744, -2.779836416244507, 0.8307797312736511, 6.8916826248168945, -1.130050778388977, -2.319080352783203, &
      3.4285099506378174, 3.0322296619415283, 9.251358985900879, 6.548008918762207, 11.802226066589355, 0.07002818584442139, &
      -6.547648906707764, 0.41237232089042664, 1.6485198736190796, -7.2986907958984375, -9.798441886901855, -0.8000574111938477, &
      5.421070098876953, 12.376362800598145, 3.5469491481781006, 0.42429041862487793, -3.4540629386901855, 6.002678871154785, &
      14.084171295166016, -6.0355544090271, -4.5586419105529785, -0.5676508545875549, -4.7427778244018555, -13.566842079162598, &
      -15.487546920776367, -10.175990104675293, -4.5678181648254395, 0.21600306034088135, -5.127737522125244, -14.529532432556152, &
      -11.48446273803711, -8.493194580078125, 2.886819839477539, -10.78671932220459, 3.0515291690826416, 3.9257333278656006, &
      1.9485543966293335, -4.58419132232666, -14.821449279785156, -0.7291865944862366, -0.6265296936035156, -3.6237568855285645, &
      12.191559791564941, -3.7630860805511475, -3.6215202808380127, -7.667520523071289, 10.962368965148926/)
   real:: w2(64) = (/1.3788957595825195, 0.6852167248725891, 0.6331512331962585, 1.022736668586731, 2.3716814517974854, &
      0.7862018346786499, 2.125812292098999, 0.44263893365859985, 1.0701590776443481, 2.0024478435516357, 0.7902749180793762, &
      -1.3123514652252197, 0.5989774465560913, 0.4642113447189331, 0.9325730800628662, 1.7831560373306274, 0.34898051619529724, &
      1.1476787328720093, 1.4043091535568237, 0.541227400302887, 1.3729044198989868, 2.4966423511505127, 1.1039432287216187, &
      0.9495366215705872, 0.3575485944747925, 0.9544470906257629, 0.25511109828948975, 1.0550057888031006, 1.0778623819351196, &
      0.8391621112823486, 1.3561758995056152, 0.6427061557769775, 0.420436829328537, 0.7205876111984253, 1.0969924926757812, &
      2.4658076763153076, 0.8278936743736267, 1.9670686721801758, 0.8705187439918518, 1.8658331632614136, -3.924913167953491, &
      0.8656800985336304, 1.686279535293579, 1.8177659511566162, 0.9218791127204895, 1.4188413619995117, -9.251812934875488, &
      -0.33778923749923706, 1.0675315856933594, 0.23395726084709167, 3.475602626800537, 0.6860080361366272, 1.0940653085708618, &
      1.3342487812042236, 0.703750729560852, 4.449444770812988, 0.7807712554931641, 0.6309940218925476, 0.5909700393676758, &
      1.6191396713256836, 0.4196356534957886, 0.5476643443107605, 1.4146426916122437, 0.9049907326698303/)
   real:: b2 = -5.665149211883545
   real:: x(6), y
   integer:: i, j
   x = (/lmr_z, par_z, rh_can, gb_mol, je, cair/)
   ! normalize
   do j = 1, 6
      x(j) = (x(j) - mean(j))/std(j)
   end do
   x0 = 0.0
   ! a 64-layer with sigmoid activation
   do i = 1, 64
      y = 0.0
      do j = 1, 6
         y = y + x(j)*w1(j, i)
      end do
      y = 1./(1.+exp(-(y + b1(i))))
      ! a 1-layer with linear activation
      x0 = x0 + w2(i)*y
   end do
   x0 = x0 + b2
end function nn2
