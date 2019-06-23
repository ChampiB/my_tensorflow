package cnn.useful.mail;

import java.util.*;
import javax.mail.*;
import javax.mail.internet.*;

public class MailHelper {

    private Properties properties = System.getProperties();
    private Session mailSession;
    private MimeMessage message;

    private void setMailServerProperties() {
        properties.put("cnn.useful.mail.smtp.host", "smtp.gmail.com");
        properties.put("cnn.useful.mail.smtp.port", "587");
        properties.put("cnn.useful.mail.smtp.auth", "true");
        properties.put("cnn.useful.mail.smtp.starttls.enable", "true");
    }

    private void createEmailMessage(String to, String subject, String body) throws AddressException, MessagingException {
        mailSession = Session.getDefaultInstance(properties, null);
        message = new MimeMessage(mailSession);
        message.addRecipient(Message.RecipientType.TO, new InternetAddress(to));
        message.setSubject(subject);
        message.setText(body);
    }

    private void sendEmail(String from) throws AddressException, MessagingException {
        String emailHost = "smtp.gmail.com";
        String fromPassword = CipherHelper.decryptFile(CipherHelper.PASSWORD_FILE_NAME);
        Transport transport = mailSession.getTransport("smtp");
        transport.connect(emailHost, from, fromPassword);
        transport.sendMessage(message, message.getAllRecipients());
        transport.close();
    }

    public void sendEmail(String to, String from, String subject, String text){
        try{
            setMailServerProperties();
            createEmailMessage(to, subject, text);
            sendEmail(from);
        }catch (MessagingException mex) {
            mex.printStackTrace();}
    }

    public static void sendTrainingIsOver() {
        String mail = CipherHelper.decryptFile(CipherHelper.EMAIL_FILE_NAME);
        String subject = "Training is over !";
        String text = "Hello, the training of your model is over !";
        new MailHelper().sendEmail(mail, mail, subject, text);
    }
}
