package cnn.useful.mail;

import org.apache.commons.io.FileUtils;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.spec.SecretKeySpec;
import java.io.File;
import java.nio.file.Files;
import java.util.Arrays;

public class CipherHelper {

    public static String KEY_FILE_NAME = getResource("key.enc");
    public static String PASSWORD_FILE_NAME = getResource("password.enc");
    public static String EMAIL_FILE_NAME = getResource("email.enc");

    private static String getResource(String resource) {
        return "./src/main/resources/" + resource;
    }

    private byte[] generateKey() throws Exception {
        KeyGenerator keyGen = KeyGenerator.getInstance("AES");
        keyGen.init(256);
        return keyGen.generateKey().getEncoded();
    }

    private byte[] loadFromFile(String fileName) throws Exception {
        File file = new File(fileName);
        return Files.readAllBytes(file.toPath());
    }

    private void saveToFile(byte[] bytes, String fileName) throws Exception {
        FileUtils.writeByteArrayToFile(new File(fileName), bytes);
    }

    private SecretKeySpec loadKey() throws Exception {
        byte[] keyBytes;
        if (new File(KEY_FILE_NAME).exists()) {
            keyBytes = loadFromFile(KEY_FILE_NAME);
        } else {
            keyBytes = generateKey();
            saveToFile(keyBytes, KEY_FILE_NAME);
        }
        return new SecretKeySpec(keyBytes, "AES");
    }

    private Cipher getCipher() throws Exception {
        return Cipher.getInstance("AES/ECB/PKCS5Padding");
    }

    private byte[] encrypt(String input) throws Exception {
        SecretKeySpec key = loadKey();
        Cipher cipher = getCipher();
        cipher.init(Cipher.ENCRYPT_MODE, key);
        byte[] encrypted= new byte[cipher.getOutputSize(input.getBytes().length)];
        int enc_len = cipher.update(input.getBytes(), 0, input.getBytes().length, encrypted, 0);
        enc_len += cipher.doFinal(encrypted, enc_len);
        return Arrays.copyOfRange(encrypted, 0, enc_len);
    }

    private String decrypt(byte[] encrypted) throws Exception {
        SecretKeySpec key = loadKey();
        Cipher cipher = getCipher();
        cipher.init(Cipher.DECRYPT_MODE, key);
        byte[] decrypted = new byte[cipher.getOutputSize(encrypted.length)];
        int dec_len = cipher.update(encrypted, 0, encrypted.length, decrypted, 0);
        dec_len += cipher.doFinal(decrypted, dec_len);
        return new String(Arrays.copyOfRange(decrypted, 0, dec_len));
    }

    public static boolean encryptFile(String input, String fileName) {
        try {
            CipherHelper cipher = new CipherHelper();
            cipher.saveToFile(cipher.encrypt(input), fileName);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    public static String decryptFile(String fileName) {
        try {
            CipherHelper cipher = new CipherHelper();
            return cipher.decrypt(cipher.loadFromFile(fileName));
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * This main create the email.enc and password.enc files.
     * @param args the first argument is the email and the second is the password to be encrypted.
     */
    public static void main(String[] args) {
        // Encrypt email and password
        System.out.println(CipherHelper.encryptFile(args[0], CipherHelper.EMAIL_FILE_NAME));
        System.out.println(CipherHelper.encryptFile(args[1], CipherHelper.PASSWORD_FILE_NAME));
        // Decrypt email and password
        System.out.println(CipherHelper.decryptFile(CipherHelper.EMAIL_FILE_NAME));
        System.out.println(CipherHelper.decryptFile(CipherHelper.PASSWORD_FILE_NAME));
    }

}
